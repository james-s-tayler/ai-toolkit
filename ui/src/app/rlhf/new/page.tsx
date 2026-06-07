'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import SessionSetupForm from '@/components/rlhf/SessionSetupForm';

export default function NewRlhfSessionPage() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (data: any) => {
    setIsSubmitting(true);
    setError('');
    try {
      const res = await apiClient.post('/api/rlhf', data);
      const sessionId = res.data.id;

      if (data.dataset_mode === 'import') {
        // Import datasets immediately after session creation
        try {
          const importRes = await apiClient.post(`/api/rlhf/${sessionId}/import`, {
            accepted_dataset: data.accepted_dataset,
            rejected_dataset: data.rejected_dataset,
          });
          const stats = importRes.data;
          console.log(
            `Import complete: ${stats.matched} matched, ` +
            `${stats.unmatched_accepted} unmatched accepted, ` +
            `${stats.unmatched_rejected} unmatched rejected`
          );
        } catch (importErr: any) {
          const importError = importErr?.response?.data?.error || 'Failed to import datasets';
          // Session was created but import failed — delete the session and show error
          try { await apiClient.delete(`/api/rlhf/${sessionId}`); } catch {}
          setError(importError);
          setIsSubmitting(false);
          return;
        }
      }

      router.push(`/rlhf/${sessionId}`);
    } catch (e: any) {
      setError(e?.response?.data?.error || 'Failed to create session');
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <TopBar>
        <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => router.back()}>
          <FaChevronLeft />
        </Button>
        <h1 className="text-lg ml-2">New RLHF Session</h1>
      </TopBar>
      <MainContent>
        <SessionSetupForm onSubmit={handleSubmit} isSubmitting={isSubmitting} error={error} />
      </MainContent>
    </>
  );
}
