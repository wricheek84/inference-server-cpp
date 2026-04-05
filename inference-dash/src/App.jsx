import React, { useRef, useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const Card = ({ title, value, unit, type = "", children }) => {
  const cardRef = useRef(null);
  const handleMouseMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    cardRef.current.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
    cardRef.current.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
  };

  return (
    <div 
      ref={cardRef}
      onMouseMove={handleMouseMove}
      className={`card ${type} p-8 flex flex-col justify-between min-h-[240px] border-white/10`}
    >
      <span className="text-gray-500 text-[10px] font-black tracking-[0.3em] uppercase">{title}</span>
      {value !== undefined ? (
        <h1 className={`text-6xl font-bold flex items-baseline ${type === 'card-p99' ? 'text-[#39FF14]' : type === 'card-p50' ? 'text-[#00FFFF]' : 'text-white'}`}>
          {value}
          <span className="text-gray-400 text-xl ml-2 font-medium lowercase italic">{unit}</span>
        </h1>
      ) : <div className="mt-4">{children}</div>}
    </div>
  );
};

function App() {
  const [stats, setStats] = useState({
    current_queue_depth: 0,
    active_workers: 0,
    total_workers: 8,
    batch_size: 32,
    timeout_ms: 25,
    tps: 0,
    total_tokens: 0,
    utilization: 0,
    p99: 0,
    p50: 0,
    total_batches: 0,
    status: "CONNECTING...",
    // --- NEW FIELDS ---
    latest_prediction: 0,
    latest_confidence: 0
  });

  const [history, setHistory] = useState([]);
  const activeTimeoutRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8080/ws');

    socket.onopen = () => setStats(prev => ({ ...prev, status: "STABLE" }));
    
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const p99_val = parseFloat(data.p99_latency || 0);
      const p50_val = parseFloat(data.p50_latency || 0);

      setHistory(prev => {
        const newHistory = [...prev, { 
          time: new Date().toLocaleTimeString().slice(-5), 
          p99: p99_val, 
          p50: p50_val 
        }];
        return newHistory.slice(-60); 
      });
      
      setStats(prev => {
        const newTotalTokens = prev.total_tokens + (data.tasks_processed || 0);
        let newActiveWorkers = prev.active_workers;

        if (data.worker_peak > 0) {
          newActiveWorkers = data.worker_peak;
          if (activeTimeoutRef.current) clearTimeout(activeTimeoutRef.current);
          activeTimeoutRef.current = setTimeout(() => {
            setStats(s => ({ ...s, active_workers: 0 }));
          }, 300);
        }

        const max_ns_per_window = 8 * 100000000;
        const util = Math.min(((data.worker_active_time_ns || 0) / max_ns_per_window) * 100, 100);

        return {
          ...prev,
          current_queue_depth: data.queue_peak || 0,
          active_workers: newActiveWorkers,
          tps: (data.tasks_processed || 0) * 10,
          utilization: util.toFixed(1),
          total_tokens: newTotalTokens,
          p99: p99_val.toFixed(1),
          p50: p50_val.toFixed(1),
          total_batches: data.total_batches || prev.total_batches,
          // --- MAPPING NEW DATA ---
          latest_prediction: data.latest_prediction,
          latest_confidence: (data.latest_confidence * 100).toFixed(2)
        };
      });
    };

    socket.onclose = () => setStats(prev => ({ ...prev, status: "OFFLINE" }));
    
    return () => {
      socket.close();
      if (activeTimeoutRef.current) clearTimeout(activeTimeoutRef.current);
    };
  }, []);

  return (
    <div className="min-h-screen bg-black text-white p-12 flex flex-col items-center font-mono">
      
      <div className="w-full max-w-[1320px] mb-12 flex justify-between items-end border-b border-white/10 pb-6">
        <div>
          <h2 className="text-gray-600 text-[10px] font-bold tracking-[0.4em] uppercase mb-1">Inference Cluster</h2>
          <h1 className="text-3xl font-black tracking-tighter">NODE_01 // <span className={stats.status === "STABLE" ? "text-[#39FF14]" : "text-red-500"}>{stats.status}</span></h1>
        </div>
        <div className="text-right">
          <h2 className="text-gray-600 text-[10px] font-bold tracking-[0.4em] uppercase mb-1">Model Architecture</h2>
          <h1 className="text-xl font-bold text-[#00FFFF] tracking-widest uppercase">DistilBERT-Quantized</h1>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-10 w-full max-w-[1320px]">
        {/* --- NEW: ADVISOR VERDICT CARD --- */}
        <div className="col-span-2">
          <Card title="SRE Advisor Verdict" type={stats.latest_prediction === 1 ? "card-red" : "card-p99"}>
            <div className="flex flex-col items-center justify-center py-4">
              <h1 className={`text-7xl font-black tracking-tighter mb-4 transition-colors duration-300 ${stats.latest_prediction === 0 ? 'text-[#39FF14]' : 'text-[#FF3131]'}`}>
                {stats.latest_prediction === 0 ? "SYSTEM_STABLE" : "ANOMALY_DETECTED"}
              </h1>
              <div className="flex items-center gap-4 bg-white/5 px-6 py-2 rounded-full border border-white/10">
                <span className="text-gray-500 text-[10px] font-black uppercase tracking-widest">Model Confidence</span>
                <span className="text-[#00FFFF] text-2xl font-bold">{stats.latest_confidence}%</span>
              </div>
            </div>
          </Card>
        </div>

        <Card title="P99 Latency" value={stats.p99} unit="ms" type="card-p99" />
        <Card title="P50 Latency" value={stats.p50} unit="ms" type="card-p50" />
        
        <Card title="Queue Depth (Peak)">
          <div className="flex flex-col gap-6">
            <div className="flex justify-between items-end">
              <span className="text-6xl font-bold text-white leading-none">{stats.current_queue_depth}</span>
              <div className="text-right leading-tight">
                <p className="text-[#00FFFF] text-[10px] font-black tracking-widest uppercase mb-1">Wait Timeout</p>
                <p className="text-2xl font-bold">{stats.timeout_ms}<span className="text-xs ml-1 text-gray-500 font-normal">ms</span></p>
              </div>
            </div>
            <div className="w-full h-4 bg-white/5 rounded-full border border-white/10 overflow-hidden relative shadow-inner">
              <div 
                className="h-full bg-[#00FFFF] shadow-[0_0_20px_rgba(0,255,255,0.4)] transition-all duration-300 ease-out" 
                style={{ width: `${Math.min((stats.current_queue_depth / stats.batch_size) * 100, 100)}%` }} 
              />
            </div>
            <p className="text-[9px] text-gray-600 font-bold uppercase tracking-[0.2em]">Batch Saturation: {stats.batch_size} Tokens Max</p>
          </div>
        </Card>
        
        <Card title="Active Workers">
          <div className="grid grid-cols-4 gap-4 mt-2">
            {[...Array(stats.total_workers)].map((_, i) => (
              <div key={i} className="flex flex-col items-center gap-2">
                <div className={`w-full h-12 rounded border-2 transition-all duration-300 ${
                  i < stats.active_workers 
                  ? 'bg-[#39FF14]/20 border-[#39FF14] shadow-[0_0_20px_rgba(57,255,20,0.5)]' 
                  : 'bg-[#FF3131]/5 border-[#FF3131]/20 shadow-none'
                }`} />
                <span className={`text-[8px] font-black ${i < stats.active_workers ? 'text-[#39FF14]' : 'text-[#FF3131]/40'}`}>
                  WK_{i+1}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* --- LIVE LATENCY GRAPH --- */}
        <div className="col-span-2">
          <Card title="Real-time Latency Stream (Last 60s)">
            <div className="h-64 w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="colorP99" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#39FF14" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#39FF14" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00FFFF" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#00FFFF" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 'auto']} stroke="#666" fontSize={10} tickFormatter={(v) => `${v}ms`} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0d0d0d', border: '1px solid #ffffff10', color: '#fff' }}
                    itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p99" 
                    stroke="#39FF14" 
                    strokeWidth={3} 
                    fillOpacity={1} 
                    fill="url(#colorP99)" 
                    isAnimationActive={false} 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p50" 
                    stroke="#00FFFF" 
                    strokeWidth={3} 
                    fillOpacity={1} 
                    fill="url(#colorP50)" 
                    isAnimationActive={false} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        <div className="col-span-2">
          <Card title="Throughput & Engine Utilization">
            <div className="grid grid-cols-3 gap-8 mt-4">
              <div className="flex flex-col items-center justify-center p-6 border border-white/5 bg-white/[0.02] rounded">
                <span className="text-[10px] text-gray-500 font-black tracking-[0.3em] uppercase mb-2">Real-time TPS</span>
                <span className="text-5xl font-bold text-white">{stats.tps}</span>
              </div>
              <div className="flex flex-col items-center justify-center p-6 border border-white/5 bg-white/[0.02] rounded">
                <span className="text-[10px] text-gray-500 font-black tracking-[0.3em] uppercase mb-2">Engine Load</span>
                <span className="text-5xl font-bold text-white">{stats.utilization}<span className="text-2xl ml-1 text-gray-500">%</span></span>
              </div>
              <div className="flex flex-col items-center justify-center p-6 border border-white/5 bg-white/[0.02] rounded">
                <span className="text-[10px] text-gray-500 font-black tracking-[0.3em] uppercase mb-2">Total Tokens</span>
                <span className="text-5xl font-bold text-white">{stats.total_tokens.toLocaleString()}</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <div className="w-full max-w-[1320px] mt-12 py-10 border-t-2 border-white/5 flex justify-between items-center">
        <div className="flex gap-16">
          <div className="flex flex-col">
            <span className="text-[10px] text-gray-600 font-black tracking-[0.4em] uppercase mb-2">Batch Velocity</span>
            <span className="text-4xl font-bold tracking-tighter">
              {stats.total_batches} <span className="text-sm text-gray-600 font-normal uppercase ml-1 tracking-widest">Total Batches</span>
            </span>
          </div>
        </div>
        
        <div className="px-8 py-3 border border-white/10 rounded-sm bg-white/[0.02] flex items-center gap-4">
          <div className={`w-3 h-3 rounded-full ${stats.status === "STABLE" ? "bg-[#39FF14] shadow-[0_0_10px_#39FF14]" : "bg-red-500 animate-pulse"}`} />
          <span className="text-[10px] text-white font-black tracking-[0.3em] uppercase">Node Status: <span className={stats.status === "STABLE" ? "text-[#39FF14]" : "text-red-500"}>{stats.status}</span></span>
        </div>
      </div>

    </div>
  );
}

export default App;