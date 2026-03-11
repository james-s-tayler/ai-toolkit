import prisma from '../prisma';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from '../paths';

const DATA_ROOT = path.join(TOOLKIT_ROOT, 'data');

async function httpGet(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith('https') ? require('https') : require('http');
    lib.get(url, (res: any) => {
      let data = '';
      res.on('data', (chunk: any) => { data += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve(data); }
      });
    }).on('error', reject);
  });
}

async function httpPost(url: string, body: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith('https') ? require('https') : require('http');
    const data = JSON.stringify(body);
    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    };
    const req = lib.request(options, (res: any) => {
      let responseData = '';
      res.on('data', (chunk: any) => { responseData += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(responseData)); }
        catch { resolve(responseData); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function downloadFile(url: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const lib = url.startsWith('https') ? require('https') : require('http');
    const dir = path.dirname(destPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const file = fs.createWriteStream(destPath);
    lib.get(url, (res: any) => {
      if (res.statusCode !== 200) {
        file.close();
        fs.unlink(destPath, () => {});
        reject(new Error(`HTTP ${res.statusCode} downloading ${url}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => { file.close(() => resolve()); });
    }).on('error', (err: Error) => { file.close(); fs.unlink(destPath, () => {}); reject(err); });
  });
}

export default async function processRlhfGeneration() {
  const sessions = await prisma.rlhfSession.findMany({ where: { status: 'generating' } });
  if (sessions.length === 0) return;

  for (const session of sessions) {
    try {
      const comfyUrl = session.comfyui_url || 'http://127.0.0.1:9188';
      const outputDir = session.output_dir || path.join(DATA_ROOT, 'rlhf', session.name);

      // Poll queued pairs
      const queuedPairs = await prisma.rlhfPair.findMany({
        where: { session_id: session.id, gen_status: 'queued' },
      });

      for (const pair of queuedPairs) {
        try {
          let aCompleted = !pair.comfyui_id_a || !!pair.image_a_path;
          let bCompleted = !pair.comfyui_id_b || !!pair.image_b_path;

          if (pair.comfyui_id_a && !pair.image_a_path) {
            const history = await httpGet(`${comfyUrl}/history/${pair.comfyui_id_a}`);
            if (history && history[pair.comfyui_id_a]?.outputs) {
              const outputs = history[pair.comfyui_id_a].outputs;
              const imageNode = Object.values(outputs).find((o: any) => o.images?.length > 0) as any;
              if (imageNode) {
                const img = imageNode.images[0];
                const imgUrl = `${comfyUrl}/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${img.type || 'output'}`;
                const destPath = path.join(outputDir, pair.id, 'image_a.png');
                await downloadFile(imgUrl, destPath);
                await prisma.rlhfPair.update({ where: { id: pair.id }, data: { image_a_path: destPath } });
                aCompleted = true;
              }
            }
          }

          if (pair.comfyui_id_b && !pair.image_b_path) {
            const history = await httpGet(`${comfyUrl}/history/${pair.comfyui_id_b}`);
            if (history && history[pair.comfyui_id_b]?.outputs) {
              const outputs = history[pair.comfyui_id_b].outputs;
              const imageNode = Object.values(outputs).find((o: any) => o.images?.length > 0) as any;
              if (imageNode) {
                const img = imageNode.images[0];
                const imgUrl = `${comfyUrl}/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${img.type || 'output'}`;
                const destPath = path.join(outputDir, pair.id, 'image_b.png');
                await downloadFile(imgUrl, destPath);
                await prisma.rlhfPair.update({ where: { id: pair.id }, data: { image_b_path: destPath } });
                bCompleted = true;
              }
            }
          }

          if (aCompleted && bCompleted) {
            await prisma.rlhfPair.update({ where: { id: pair.id }, data: { gen_status: 'completed' } });
          }
        } catch (err) {
          console.error(`[rlhf] Error polling pair ${pair.id}:`, err);
        }
      }

      // Submit pending pairs (up to 5 at a time)
      const BATCH_SIZE = 5;
      const alreadyQueued = await prisma.rlhfPair.count({ where: { session_id: session.id, gen_status: 'queued' } });
      const slotsAvailable = BATCH_SIZE - alreadyQueued;

      const workflowTemplate = session.workflow_json;
      if (slotsAvailable > 0 && workflowTemplate) {
        const pendingPairs = await prisma.rlhfPair.findMany({
          where: { session_id: session.id, gen_status: 'pending' },
          take: slotsAvailable,
        });

        for (const pair of pendingPairs) {
          try {
            // Escape the prompt for safe JSON template substitution.
            // JSON.stringify wraps the value in quotes and escapes special chars.
            // We slice off the surrounding quotes since {{PROMPT}} is already
            // inside a JSON string literal in the workflow template.
            const escapedPrompt = JSON.stringify(pair.prompt).slice(1, -1);
            const workflowA = workflowTemplate.replace(/\{\{PROMPT\}\}/g, escapedPrompt).replace(/\{\{SEED\}\}/g, String(pair.seed_a));
            const workflowB = workflowTemplate.replace(/\{\{PROMPT\}\}/g, escapedPrompt).replace(/\{\{SEED\}\}/g, String(pair.seed_b));

            let parsedA: any;
            let parsedB: any;
            try {
              parsedA = JSON.parse(workflowA);
              parsedB = JSON.parse(workflowB);
            } catch (parseErr) {
              console.error(`[rlhf] Invalid workflow JSON for pair ${pair.id}:`, parseErr);
              await prisma.rlhfPair.update({ where: { id: pair.id }, data: { gen_status: 'error' } });
              continue;
            }
            const resA = await httpPost(`${comfyUrl}/prompt`, { prompt: parsedA });
            const comfyui_id_a = resA.prompt_id || '';
            const resB = await httpPost(`${comfyUrl}/prompt`, { prompt: parsedB });
            const comfyui_id_b = resB.prompt_id || '';

            await prisma.rlhfPair.update({
              where: { id: pair.id },
              data: { gen_status: 'queued', comfyui_id_a, comfyui_id_b },
            });
          } catch (err) {
            console.error(`[rlhf] Error submitting pair ${pair.id}:`, err);
            await prisma.rlhfPair.update({ where: { id: pair.id }, data: { gen_status: 'error' } });
          }
        }
      }

      // Check if all pairs are completed
      const totalPairs = await prisma.rlhfPair.count({ where: { session_id: session.id } });
      const completedPairs = await prisma.rlhfPair.count({ where: { session_id: session.id, gen_status: 'completed' } });
      if (totalPairs > 0 && completedPairs === totalPairs) {
        await prisma.rlhfSession.update({ where: { id: session.id }, data: { status: 'generated' } });
      }
    } catch (err) {
      console.error(`[rlhf] Error processing session ${session.id}:`, err);
    }
  }
}
