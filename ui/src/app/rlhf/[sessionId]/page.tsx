'use client';

import { useState, useEffect, useCallback, useMemo, useRef, use } from 'react';
import { useRouter } from 'next/navigation';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import { RlhfSession, RlhfTrainingRun } from '@prisma/client';
import GenerationMonitor from '@/components/rlhf/GenerationMonitor';
import EvaluationUI from '@/components/rlhf/EvaluationUI';
import TrainingConfig from '@/components/rlhf/TrainingConfig';
import TrainingMonitor from '@/components/rlhf/TrainingMonitor';
import GPUWidget from '@/components/GPUWidget';
import CPUWidget from '@/components/CPUWidget';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';

type TabKey = 'generation' | 'evaluation' | 'training' | 'pairs';

const tabs: { key: TabKey; label: string }[] = [
  { key: 'generation', label: 'Generation' },
  { key: 'evaluation', label: 'Evaluation' },
  { key: 'training', label: 'Training' },
  { key: 'pairs', label: 'Pairs' },
];

interface RlhfPair {
  id: string;
  prompt: string;
  gen_status: string;
  preference: string;
  image_a_path: string;
  image_b_path: string;
  created_at: string;
}

const clean = (text: string): string => {
  return text.replace(/\x1B\[A/g, '');
};

export default function SessionPage({ params }: { params: { sessionId: string } }) {
  const usableParams = use(params as any) as { sessionId: string };
  const sessionId = usableParams.sessionId;
  const router = useRouter();
  const [session, setSession] = useState<RlhfSession | null>(null);
  const [tab, setTab] = useState<TabKey>('generation');
  const [isTrainingStarting, setIsTrainingStarting] = useState(false);
  const [trainingError, setTrainingError] = useState('');
  const [evaluatedCount, setEvaluatedCount] = useState(0);
  const [pairs, setPairs] = useState<RlhfPair[]>([]);
  const [pairsTotal, setPairsTotal] = useState(0);
  const [pairsPage, setPairsPage] = useState(1);
  const [pairsFilter, setPairsFilter] = useState('');
  const [prefFilter, setPrefFilter] = useState('');
  const [isLoadingPairs, setIsLoadingPairs] = useState(false);

  // Training log state
  const [trainingLog, setTrainingLog] = useState('');
  const [latestRunId, setLatestRunId] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  // GPU/CPU monitoring
  const gpuIds = useMemo(() => {
    if (!session?.gpu_ids) return null;
    return session.gpu_ids.split(',').map(id => parseInt(id));
  }, [session?.gpu_ids]);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, tab === 'training' ? 5000 : null);
  const { cpuInfo, isCPUInfoLoaded } = useCPUInfo(tab === 'training' ? 5000 : null);

  const fetchSession = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}`);
      setSession(res.data);
    } catch (e) {
      console.error('Error fetching session:', e);
    }
  }, [sessionId]);

  const fetchEvaluatedCount = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/evaluate`);
      setEvaluatedCount(res.data.stats?.evaluated ?? 0);
    } catch (e) {
      // ignore
    }
  }, [sessionId]);

  const fetchPairs = useCallback(async (page = 1, genStatus = '', pref = '') => {
    setIsLoadingPairs(true);
    try {
      let url = `/api/rlhf/${sessionId}/pairs?page=${page}&perPage=50`;
      if (genStatus) url += `&gen_status=${genStatus}`;
      if (pref) url += `&preference=${pref}`;
      const res = await apiClient.get(url);
      setPairs(res.data.pairs);
      setPairsTotal(res.data.total);
    } catch (e) {
      console.error('Error fetching pairs:', e);
    } finally {
      setIsLoadingPairs(false);
    }
  }, [sessionId]);

  // Fetch latest run ID for log polling
  const fetchLatestRun = useCallback(async () => {
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/train`);
      const runs: RlhfTrainingRun[] = res.data.runs;
      if (runs.length > 0) {
        setLatestRunId(runs[0].id);
      }
    } catch (e) {
      // ignore
    }
  }, [sessionId]);

  // Fetch training log
  const fetchLog = useCallback(async () => {
    if (!latestRunId) return;
    try {
      const res = await apiClient.get(`/api/rlhf/${sessionId}/train/${latestRunId}/log`);
      if (res.data.log) {
        setTrainingLog(clean(res.data.log));
      }
    } catch (e) {
      // ignore
    }
  }, [sessionId, latestRunId]);

  useEffect(() => { fetchSession(); fetchEvaluatedCount(); }, [fetchSession, fetchEvaluatedCount]);

  // Re-fetch evaluated count when switching to training tab so it's always current
  useEffect(() => {
    if (tab === 'training') {
      fetchEvaluatedCount();
      fetchLatestRun();
    }
  }, [tab, fetchEvaluatedCount, fetchLatestRun]);

  // Poll training log every 2s when on training tab
  useEffect(() => {
    if (tab !== 'training' || !latestRunId) return;
    fetchLog();
    const timer = setInterval(fetchLog, 2000);
    return () => clearInterval(timer);
  }, [tab, latestRunId, fetchLog]);

  useEffect(() => {
    if (tab === 'pairs') fetchPairs(pairsPage, pairsFilter, prefFilter);
  }, [tab, pairsPage, pairsFilter, prefFilter, fetchPairs]);

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [trainingLog, isScrolledToBottom]);

  const handleLogScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      setIsScrolledToBottom(scrollHeight - scrollTop - clientHeight < 10);
    }
  };

  const handleStartTraining = async (params: any) => {
    setIsTrainingStarting(true);
    setTrainingError('');
    try {
      await apiClient.post(`/api/rlhf/${sessionId}/train`, params);
      await fetchSession();
      await fetchLatestRun();
      setTab('training');
    } catch (e: any) {
      setTrainingError(e?.response?.data?.error || 'Failed to start training');
    } finally {
      setIsTrainingStarting(false);
    }
  };

  const logLines = useMemo(() => {
    let splits = trainingLog.split(/\n|\r\n/);
    splits = splits.map(line => line.split(/\r/).pop()) as string[];
    const maxLines = 1000;
    if (splits.length > maxLines) {
      splits = splits.slice(splits.length - maxLines);
    }
    return splits;
  }, [trainingLog]);

  const prefLabels: Record<string, string> = { none: '\u2014', a: 'A', b: 'B', tie: 'Tie', skip: 'Skip' };
  const prefColors: Record<string, string> = {
    none: 'text-gray-500', a: 'text-green-400', b: 'text-blue-400', tie: 'text-yellow-400', skip: 'text-gray-500'
  };
  const statusColors: Record<string, string> = {
    pending: 'text-gray-500', queued: 'text-yellow-400', completed: 'text-green-400', error: 'text-red-400'
  };

  return (
    <>
      <TopBar>
        <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => router.push('/rlhf')}>
          <FaChevronLeft />
        </Button>
        <div>
          <h1 className="text-lg">
            {session?.name ?? 'Loading...'}
            {session && <span className="ml-3 text-sm text-gray-500">{session.status}</span>}
          </h1>
        </div>
      </TopBar>

      {/* Tab bar */}
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-2 text-sm z-10">
        {tabs.map(t => (
          <Button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-1 h-8 ${t.key === tab ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-gray-200'}`}
          >
            {t.label}
          </Button>
        ))}
      </div>

      <MainContent className="pt-20">
        {!session && <p className="text-gray-400">Loading...</p>}
        {session && (
          <>
            {tab === 'generation' && (
              <GenerationMonitor
                sessionId={sessionId}
                sessionStatus={session.status}
                onStatusChange={fetchSession}
              />
            )}

            {tab === 'evaluation' && (
              <EvaluationUI sessionId={sessionId} sessionStatus={session.status} />
            )}

            {tab === 'training' && (
              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                {/* Left 2 columns: training status + console log + config */}
                <div className="col-span-2 space-y-6">
                  {(session.status === 'training' || session.status === 'completed') && (
                    <TrainingMonitor sessionId={sessionId} onStatusChange={fetchSession} />
                  )}

                  {/* Console log */}
                  {latestRunId && (
                    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
                      <div className="bg-gray-800 px-4 py-2">
                        <h3 className="text-sm text-gray-300">Console Output</h3>
                      </div>
                      <div className="relative min-h-60 h-80">
                        <div
                          ref={logRef}
                          className="text-xs text-gray-300 absolute inset-0 p-4 overflow-y-auto"
                          onScroll={handleLogScroll}
                        >
                          <div>
                            {logLines.map((line, index) => (
                              <pre key={index}>{line}</pre>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <TrainingConfig
                    onStart={handleStartTraining}
                    isStarting={isTrainingStarting}
                    error={trainingError}
                    evaluatedCount={evaluatedCount}
                  />
                </div>

                {/* Right column: CPU + GPU widgets */}
                <div className="col-span-1">
                  <div>{isCPUInfoLoaded && cpuInfo && <CPUWidget cpu={cpuInfo} />}</div>
                  <div className="mt-4">
                    {isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}
                  </div>
                </div>
              </div>
            )}

            {tab === 'pairs' && (
              <div className="space-y-4">
                {/* Filters */}
                <div className="flex gap-3 items-end">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Generation Status</label>
                    <select
                      value={pairsFilter}
                      onChange={e => { setPairsFilter(e.target.value); setPairsPage(1); }}
                      className="text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm"
                    >
                      <option value="">All</option>
                      <option value="pending">Pending</option>
                      <option value="queued">Queued</option>
                      <option value="completed">Completed</option>
                      <option value="error">Error</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Preference</label>
                    <select
                      value={prefFilter}
                      onChange={e => { setPrefFilter(e.target.value); setPairsPage(1); }}
                      className="text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm"
                    >
                      <option value="">All</option>
                      <option value="none">Not evaluated</option>
                      <option value="a">A wins</option>
                      <option value="b">B wins</option>
                      <option value="tie">Tie</option>
                      <option value="skip">Skip</option>
                    </select>
                  </div>
                  <span className="text-gray-500 text-sm">{pairsTotal} pairs</span>
                </div>

                {isLoadingPairs && <p className="text-gray-400 text-sm">Loading pairs...</p>}

                {/* Pairs grid */}
                <div className="grid grid-cols-1 gap-2">
                  {pairs.map(pair => (
                    <div key={pair.id} className="bg-gray-900 rounded-lg p-3 flex gap-4 items-start">
                      {/* Thumbnail pair */}
                      <div className="flex gap-2 flex-shrink-0">
                        {pair.image_a_path ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={`/api/img/${encodeURIComponent(pair.image_a_path)}`} alt="A" className="w-16 h-16 object-cover rounded bg-gray-800" />
                        ) : (
                          <div className="w-16 h-16 rounded bg-gray-800 flex items-center justify-center text-gray-600 text-xs">A</div>
                        )}
                        {pair.image_b_path ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={`/api/img/${encodeURIComponent(pair.image_b_path)}`} alt="B" className="w-16 h-16 object-cover rounded bg-gray-800" />
                        ) : (
                          <div className="w-16 h-16 rounded bg-gray-800 flex items-center justify-center text-gray-600 text-xs">B</div>
                        )}
                      </div>
                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-300 text-sm truncate">{pair.prompt}</p>
                        <div className="flex gap-4 mt-1 text-xs">
                          <span className={statusColors[pair.gen_status] ?? 'text-gray-500'}>{pair.gen_status}</span>
                          <span className={prefColors[pair.preference] ?? 'text-gray-500'}>
                            {prefLabels[pair.preference] ?? pair.preference}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Pagination */}
                {pairsTotal > 50 && (
                  <div className="flex gap-2 items-center justify-center pt-2">
                    <button
                      disabled={pairsPage === 1}
                      onClick={() => setPairsPage(p => p - 1)}
                      className="text-gray-400 hover:text-gray-200 px-3 py-1 disabled:opacity-40"
                    >
                      &larr; Prev
                    </button>
                    <span className="text-gray-500 text-sm">Page {pairsPage} of {Math.ceil(pairsTotal / 50)}</span>
                    <button
                      disabled={pairsPage >= Math.ceil(pairsTotal / 50)}
                      onClick={() => setPairsPage(p => p + 1)}
                      className="text-gray-400 hover:text-gray-200 px-3 py-1 disabled:opacity-40"
                    >
                      Next &rarr;
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </MainContent>
    </>
  );
}
