import { useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { SimulationData } from "../useSimulationData";

const emotionColor: Record<string, string> = {
  joy: "#facc15",
  fear: "#f97373",
  sadness: "#60a5fa",
  anger: "#fb7185",
  surprise: "#a855f7",
  disgust: "#4ade80",
  neutral: "#e5e7eb",
};

interface DreamSceneProps {
  simulation: SimulationData;
}

export function DreamScene({ simulation }: DreamSceneProps) {
  const { segments, summary } = simulation;
  const topNodes = summary.memory.top_nodes ?? [];

  const sphereData = useMemo(() => {
    const nodes = topNodes.length ? topNodes : [];
    if (!nodes.length) return [] as any[];

    return nodes.map((node, index) => {
      const angle = (index / nodes.length) * Math.PI * 2;
      const radius = 6 + index * 0.4;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = 0.5 * Math.sin(angle * 2);
      return {
        id: node.id,
        label: node.label,
        emotion: node.emotion,
        count: node.count,
        salience: node.salience,
        position: new THREE.Vector3(x, y, z),
      };
    });
  }, [topNodes]);

  const totalTime = segments.length
    ? segments[segments.length - 1].end_time_hours - segments[0].start_time_hours
    : 1;

  useFrame((state, delta) => {
    const t = (state.clock.getElapsedTime() * 0.05) % 1; // normalized 0-1
    const theta = t * Math.PI * 2;
    const radius = 14;
    const camX = Math.cos(theta) * radius;
    const camZ = Math.sin(theta) * radius;
    const camY = 6 + 2 * Math.sin(theta * 2);

    state.camera.position.lerp(new THREE.Vector3(camX, camY, camZ), delta * 0.5);
    state.camera.lookAt(0, 0, 0);
  });

  return (
    <group>
      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshStandardMaterial color="#020617" roughness={0.8} metalness={0.2} />
      </mesh>

      {/* Time ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[9.5, 10, 128]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} />
      </mesh>

      {/* Memory nodes as spheres */}
      {sphereData.map((node) => {
        const color = emotionColor[node.emotion] ?? emotionColor.neutral;
        const scale = 0.6 + 0.2 * Math.log2(1 + node.count);
        return (
          <group key={node.id} position={node.position}>
            <mesh>
              <sphereGeometry args={[scale, 32, 32]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.8}
                metalness={0.6}
                roughness={0.2}
              />
            </mesh>
            <Html distanceFactor={12} center style={{ pointerEvents: "none" }}>
              <div
                style={{
                  padding: "2px 6px",
                  borderRadius: "999px",
                  background: "rgba(15,23,42,0.85)",
                  border: "1px solid rgba(148,163,184,0.6)",
                  fontSize: "10px",
                  whiteSpace: "nowrap",
                }}
              >
                {node.label}
              </div>
            </Html>
          </group>
        );
      })}

      {/* Subtle info overlay */}
      <Html position={[0, 4.5, 0]} center>
        <div
          style={{
            padding: "4px 10px",
            borderRadius: "999px",
            background: "rgba(15,23,42,0.9)",
            border: "1px solid rgba(148,163,184,0.6)",
            fontSize: "11px",
          }}
        >
          Total segments: {segments.length} · Night span: {totalTime.toFixed(2)} h
        </div>
      </Html>
    </group>
  );
}
