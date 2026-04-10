import { useState } from "react";
import axios from "axios";

export interface DreamSegment {
  id: string;
  start_time_hours: number;
  end_time_hours: number;
  stage: string;
  narrative: string;
  scene_description: string;
  dominant_emotion: string;
  bizarreness_score: number;
  lucidity_probability: number;
}

export interface SimulationSummary {
  sleep_stages: Record<string, number>;
  neurochemistry: Record<string, { mean: number; max: number }>;
  bizarreness: { mean: number; std: number; top_segments: any[] };
  memory: { top_nodes: { id: string; label: string; emotion: string; count: number; salience: number }[] };
}

export interface SimulationData {
  id: string;
  segments: DreamSegment[];
  summary: SimulationSummary;
}

export function useSimulationData(setLoading: (v: boolean) => void) {
  const [data, setData] = useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async () => {
    try {
      setError(null);
      setLoading(true);
      const body = {
        duration_hours: 8.0,
        dt_minutes: 0.5,
        ssri_strength: 1.0,
        stress_level: 0.2,
        llm_enabled: false,
        llm_provider: null,
        llm_model: null,
        llm_important_only: true,
      };
      const res = await axios.post("/api/simulation/night", body);
      const payload = res.data;
      setData({ id: payload.id, segments: payload.segments, summary: payload.summary });
    } catch (err: any) {
      setError(err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return { data, error, runSimulation };
}
