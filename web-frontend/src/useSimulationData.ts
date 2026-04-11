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
  config: Record<string, any>;
}

export interface SimulationConfig {
  durationHours: number;
  dtMinutes: number;
  ssriStrength: number;
  stressLevel: number;
  llmEnabled: boolean;
  llmProvider: string | null;
  llmModel: string | null;
  llmImportantOnly: boolean;
  llmApiKey: string | null;
}

export function useSimulationData(
  setLoading: (v: boolean) => void,
  setProgress?: (v: number) => void,
) {
  const [data, setData] = useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async (config: SimulationConfig) => {
    let timer: number | undefined;

    try {
      setError(null);
      setLoading(true);
      if (setProgress) {
        setProgress(0);
        let value = 0;
        timer = window.setInterval(() => {
          value = Math.min(90, value + 5 + Math.random() * 10);
          setProgress(Math.floor(value));
        }, 260);
      }

      const body = {
        duration_hours: config.durationHours,
        dt_minutes: config.dtMinutes,
        ssri_strength: config.ssriStrength,
        stress_level: config.stressLevel,
        llm_enabled: config.llmEnabled,
        llm_provider: config.llmProvider,
        llm_model: config.llmModel,
        llm_api_key: config.llmApiKey,
        llm_important_only: config.llmImportantOnly,
      };

      const res = await axios.post("/simulate-night", body);
      const payload = res.data;
      setData({
        id: payload.id,
        segments: payload.segments,
        summary: payload.summary,
        config: payload.config ?? {},
      });
    } catch (err: any) {
      setError(err?.message ?? "Unknown error");
    } finally {
      if (timer !== undefined) {
        window.clearInterval(timer);
      }
      if (setProgress) {
        setProgress(100);
        setTimeout(() => setProgress(0), 600);
      }
      setLoading(false);
    }
  };

  return { data, error, runSimulation };
}
