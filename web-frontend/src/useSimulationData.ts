import { useState } from "react";
import axios from "axios";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

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
  memory: {
    top_nodes: {
      id: string;
      label: string;
      emotion: string;
      count: number;
      salience: number;
    }[];
  };
}

export interface SleepState {
  time_hours: number;
  stage: string;
  process_s: number;
  process_c: number;
}

export interface NeuroState {
  time_hours: number;
  ach: number;
  serotonin: number;
  ne: number;
  cortisol: number;
}

export interface MemoryNode {
  id: string;
  label: string;
  emotion?: string;
  salience?: number;
  activation?: number;
  [key: string]: any;
}

export interface MemoryEdge {
  source: string;
  target: string;
  weight?: number;
  [key: string]: any;
}

export interface MemoryGraph {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
}

export interface SimulationData {
  id: string;
  segments: DreamSegment[];
  summary: SimulationSummary;
  sleep_history: SleepState[];
  neuro_history: NeuroState[];
  memory_graph: MemoryGraph;
}

// ---------------------------------------------------------------------------
// Request config
// ---------------------------------------------------------------------------

export interface SimulationRequestConfig {
  duration_hours: number;
  dt_minutes: number;
  ssri_strength: number;
  stress_level: number;
  llm_enabled: boolean;
  llm_provider: string | null;
  llm_model: string | null;
  llm_important_only: boolean;
  llm_api_key: string | null;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSimulationData(setLoading: (v: boolean) => void) {
  const [data, setData] = useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async (config: SimulationRequestConfig) => {
    try {
      setError(null);
      setLoading(true);
      const res = await axios.post("/api/simulation/night", config);
      const payload = res.data;
      setData({
        id: payload.id,
        segments: payload.segments,
        summary: payload.summary,
        sleep_history: payload.sleep_history ?? [],
        neuro_history: payload.neuro_history ?? [],
        memory_graph: payload.memory_graph ?? { nodes: [], edges: [] },
      });
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return { data, error, runSimulation };
}
