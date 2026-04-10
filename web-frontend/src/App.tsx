import { Suspense, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { DreamScene } from "./components/DreamScene";
import { useSimulationData } from "./useSimulationData";

export function App() {
  const [loading, setLoading] = useState(false);
  const { data, error, runSimulation } = useSimulationData(setLoading);

  return (
    <div style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          padding: "0.75rem 1.5rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "linear-gradient(90deg,#020617,#0f172a)",
          borderBottom: "1px solid rgba(148,163,184,0.5)",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "1.1rem" }}>DreamForge 3D Dream Space</h1>
          <p style={{ margin: 0, fontSize: "0.75rem", opacity: 0.8 }}>
            3D visualization of simulated dreams: memory nodes as stars, camera path as timeline.
          </p>
        </div>
        <button
          onClick={runSimulation}
          style={{
            padding: "0.4rem 0.9rem",
            borderRadius: "999px",
            border: "none",
            background:
              "radial-gradient(circle at 0 0,#22d3ee 0,#0ea5e9 35%,#4f46e5 70%,#0f172a 100%)",
            color: "white",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          {loading ? "Simulating..." : "Run Simulation"}
        </button>
      </header>
      <main style={{ flex: 1 }}>
        {error && (
          <div style={{ padding: "0.75rem 1.5rem", color: "#fecaca", fontSize: "0.8rem" }}>
            Error: {error}
          </div>
        )}
        <Canvas camera={{ position: [0, 4, 10], fov: 55 }}>
          <color attach="background" args={["#020617"]} />
          <fog attach="fog" args={["#020617", 10, 40]} />
          <ambientLight intensity={0.2} />
          <directionalLight intensity={0.8} position={[5, 10, 5]} />
          <Stars radius={80} depth={50} count={5000} factor={4} fade speed={1} />
          <Suspense fallback={null}>
            {data && <DreamScene simulation={data} />}
          </Suspense>
          <OrbitControls enablePan={false} minDistance={5} maxDistance={40} />
        </Canvas>
      </main>
    </div>
  );
}
