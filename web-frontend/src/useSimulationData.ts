import { useCallback, useEffect, useRef, useState } from "react";

export interface SimulateRequest {
  duration_hours: number;
  dt_minutes: number;
  sleep_start_hour: number;
  ssri_strength: number;
  stress_level: number;
  melatonin: boolean;
  cannabis: boolean;
  prior_day_events: string[];
  emotional_state: string;
  style_preset: string;
  prompt_profile: "A" | "B";
  use_llm: boolean;
  llm_segments_only: boolean;
}

export interface DreamSegment {
  id?: string;
  segment_index?: number;
  start_time_hours?: number;
  end_time_hours?: number;
  time_hours?: number;
  stage: string;
  narrative: string;
  dominant_emotion: string;
  bizarreness_score: number;
  lucidity_probability: number;
  active_memory_ids: string[];
}

export interface SimulationResult {
  id: string;
  config: SimulateRequest;
  segments: DreamSegment[];
  summary: Record<string, unknown>;
  memory_graph: { nodes: unknown[]; edges: unknown[] };
  neurochemistry_series?: Record<string, unknown>[];
}

export interface JobStatusResponse {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelling" | "cancelled";
  phase?: string;
  progress_percent?: number;
  eta_seconds?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  simulation_id?: string | null;
}

export type SimStatus = "idle" | "running" | "complete" | "error" | "cancelling";

export interface SimulationData {
  segments: Array<{ start_time_hours: number; end_time_hours: number }>;
  summary: { memory: { top_nodes?: Array<{ id: string; label: string; emotion: string; count: number; salience: number }> } };
}

export function useSimulationData() {
  const [status, setStatus] = useState<SimStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [currentStage, setCurrentStage] = useState("");
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [liveSegments, setLiveSegments] = useState<DreamSegment[]>([]);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const pollingRef = useRef<number | null>(null);
  const statusRef = useRef<SimStatus>("idle");

  const _stopPolling = () => {
    if (pollingRef.current !== null) {
      window.clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const _cancelJobBestEffort = useCallback((jobId: string) => {
    const url = `/api/simulation/jobs/${jobId}/cancel`;
    const payload = "{}";
    try {
      if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
        const blob = new Blob([payload], { type: "application/json" });
        navigator.sendBeacon(url, blob);
        return;
      }
    } catch {
      // Fall through to keepalive fetch.
    }
    void fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      keepalive: true,
    }).catch(() => {
      // Ignore network errors during unload cancellation.
    });
  }, []);

  const _pollJob = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`/api/simulation/jobs/${jobId}`);
      if (!response.ok) {
        throw new Error(`Job polling failed: ${response.status}`);
      }
      const job: JobStatusResponse = await response.json();
      const pct = Math.max(0, Math.min(100, Number(job.progress_percent ?? 0)));
      setProgress(pct / 100);
      setCurrentStage(String(job.phase ?? job.status).toUpperCase());
      setProgressMsg(`Status: ${job.status} · ${pct.toFixed(1)}%`);

      if (job.status === "pending" || job.status === "running" || job.status === "cancelling") {
        pollingRef.current = window.setTimeout(() => {
          void _pollJob(jobId);
        }, 1000);
        return;
      }

      if (job.status === "completed" && job.simulation_id) {
        const resultResp = await fetch(`/api/simulation/${job.simulation_id}`);
        if (!resultResp.ok) {
          throw new Error(`Result fetch failed: ${resultResp.status}`);
        }
        const payload: SimulationResult = await resultResp.json();
        setResult(payload);
        setLiveSegments(payload.segments ?? []);
        setProgress(1.0);
        setProgressMsg("Simulation complete.");
        setStatus("complete");
        activeJobIdRef.current = null;
        return;
      }

      if (job.status === "cancelled") {
        setStatus("idle");
        setProgressMsg("Simulation cancelled.");
        activeJobIdRef.current = null;
        return;
      }

      if (job.status === "failed") {
        setError(job.error_message ?? job.error_code ?? "Simulation failed");
        setStatus("error");
        activeJobIdRef.current = null;
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatus("error");
      activeJobIdRef.current = null;
    }
  }, []);

  const runSimulation = useCallback(
    async (req: SimulateRequest) => {
      _stopPolling();
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      setStatus("running");
      setProgress(0);
      setCurrentStage("QUEUED");
      setProgressMsg("Queueing simulation...");
      setLiveSegments([]);
      setResult(null);
      setError(null);

      try {
        const asyncResp = await fetch("/api/simulation/night/async", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
          signal: abortRef.current.signal,
        });

        if (asyncResp.ok) {
          const submit = await asyncResp.json();
          const jobId = String(submit.job_id ?? "");
          if (!jobId) {
            throw new Error("Async simulation returned no job_id");
          }
          activeJobIdRef.current = jobId;
          setProgressMsg(`Simulation queued: ${jobId.slice(0, 8)}...`);
          await _pollJob(jobId);
          return;
        }

        const syncResp = await fetch("/api/simulation/night", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
          signal: abortRef.current.signal,
        });
        if (!syncResp.ok) {
          const text = await syncResp.text();
          throw new Error(`API error ${syncResp.status}: ${text}`);
        }
        const payload: SimulationResult = await syncResp.json();
        setResult(payload);
        setLiveSegments(payload.segments ?? []);
        setProgress(1.0);
        setCurrentStage("COMPLETE");
        setProgressMsg("Simulation complete.");
        setStatus("complete");
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unknown simulation error";
        if (message.includes("AbortError")) {
          setStatus("idle");
          setProgressMsg("Simulation cancelled.");
        } else {
          setError(message);
          setStatus("error");
        }
      }
    },
    [_pollJob]
  );

  const cancel = useCallback(async () => {
    abortRef.current?.abort();
    const jobId = activeJobIdRef.current;
    if (!jobId) {
      setStatus("idle");
      setProgressMsg("Simulation cancelled.");
      return;
    }
    setStatus("cancelling");
    setProgressMsg("Cancelling simulation...");
    try {
      await fetch(`/api/simulation/jobs/${jobId}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
    } catch {
      // Keep local cancellation state even if network cancellation fails.
    }
  }, []);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    const handleUnload = () => {
      const jobId = activeJobIdRef.current;
      const isActive =
        statusRef.current === "running" || statusRef.current === "cancelling";
      if (!jobId || !isActive) {
        return;
      }
      abortRef.current?.abort();
      _stopPolling();
      _cancelJobBestEffort(jobId);
    };

    window.addEventListener("beforeunload", handleUnload);
    window.addEventListener("pagehide", handleUnload);
    return () => {
      window.removeEventListener("beforeunload", handleUnload);
      window.removeEventListener("pagehide", handleUnload);
    };
  }, [_cancelJobBestEffort]);

  return {
    status,
    progress,
    progressMsg,
    currentStage,
    result,
    liveSegments,
    error,
    runSimulation,
    cancel,
  };
}
