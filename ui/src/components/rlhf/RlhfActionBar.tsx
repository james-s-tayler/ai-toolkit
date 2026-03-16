'use client';

import { useState } from 'react';
import { Play, Pause, Trash2, Cog } from 'lucide-react';
import { Button } from '@headlessui/react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { apiClient } from '@/utils/api';
import { TrainingParams } from '@/components/rlhf/TrainingConfig';

interface Props {
  sessionId: string;
  latestRunId: string | null;
  runStatus: string | null;
  savedConfig: TrainingParams | null;
  evaluatedCount: number;
  onTrainingStarted: () => void;
  onActionComplete: () => void;
  onRunDeleted: () => void;
}

export default function RlhfActionBar({
  sessionId,
  latestRunId,
  runStatus,
  savedConfig,
  evaluatedCount,
  onTrainingStarted,
  onActionComplete,
  onRunDeleted,
}: Props) {
  const [isStarting, setIsStarting] = useState(false);

  const isRunning = runStatus === 'running';
  const isPaused = runStatus === 'paused';
  const isActive = isRunning || isPaused;
  const canStart = !isActive && !isStarting && evaluatedCount > 0;
  const canStop = isRunning;

  const handleStart = async () => {
    if (!savedConfig) {
      openConfirm({
        title: 'Cannot Start Training',
        message: 'No training config saved. Go to the Config tab to configure and save parameters.',
        type: 'info',
        confirmText: 'OK',
      });
      return;
    }
    setIsStarting(true);
    try {
      await apiClient.post(`/api/rlhf/${sessionId}/train`, savedConfig);
      onTrainingStarted();
    } catch (e: any) {
      console.error('Failed to start training:', e);
    } finally {
      setIsStarting(false);
    }
  };

  const handleResume = async () => {
    if (!latestRunId) return;
    try {
      await apiClient.put(`/api/rlhf/${sessionId}/train/${latestRunId}`, { action: 'resume' });
      onActionComplete();
    } catch (e: any) {
      console.error('Failed to resume training:', e);
    }
  };

  const handleAction = async (action: 'pause' | 'stop') => {
    if (!latestRunId) return;
    try {
      await apiClient.put(`/api/rlhf/${sessionId}/train/${latestRunId}`, { action });
      onActionComplete();
    } catch (e: any) {
      console.error(`Failed to ${action} training:`, e);
    }
  };

  const handleMarkStopped = () => {
    if (!latestRunId) return;
    openConfirm({
      title: 'Mark as Stopped',
      message: `Are you sure you want to mark this training run as stopped? This will set the status to 'stopped' if it is hung. Only do this if you are 100% sure the training process is no longer running. This will NOT stop the process.`,
      type: 'warning',
      confirmText: 'Mark as Stopped',
      onConfirm: async () => {
        try {
          await apiClient.put(`/api/rlhf/${sessionId}/train/${latestRunId}`, { action: 'mark_stopped' });
          onActionComplete();
        } catch (e: any) {
          console.error('Failed to mark as stopped:', e);
        }
      },
    });
  };

  const handleDeleteRun = () => {
    if (!latestRunId) return;
    let message = 'Delete this training run? This cannot be undone.';
    if (isActive) {
      message += ' WARNING: The training is currently active. You should stop it first if you can.';
    }
    openConfirm({
      title: 'Delete Training Run',
      message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          await apiClient.delete(`/api/rlhf/${sessionId}/train/${latestRunId}`);
          onRunDeleted();
        } catch (e: any) {
          console.error('Failed to delete training run:', e);
        }
      },
    });
  };

  return (
    <div className={`flex items-center`}>
      {canStart && (
        <Button
          onClick={handleStart}
          disabled={!canStart}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {isPaused && (
        <Button
          onClick={handleResume}
          className={`ml-2 opacity-100`}
        >
          <Play />
        </Button>
      )}
      {canStop && (
        <Button
          onClick={() => {
            openConfirm({
              title: 'Pause Training',
              message: 'Are you sure you want to pause the training? You CAN resume later.',
              type: 'info',
              confirmText: 'Pause',
              onConfirm: async () => {
                await handleAction('pause');
              },
            });
          }}
          className={`ml-2 opacity-100`}
        >
          <Pause />
        </Button>
      )}
      {latestRunId && (
        <Button
          onClick={handleDeleteRun}
          className={`ml-2 opacity-100`}
        >
          <Trash2 />
        </Button>
      )}
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-2'}>
          <Cog />
        </MenuButton>
        <MenuItems anchor="bottom" className="z-50 bg-gray-900 border border-gray-700 rounded shadow-lg w-48 px-2 py-2 mt-4">
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded"
              onClick={handleMarkStopped}
            >
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
