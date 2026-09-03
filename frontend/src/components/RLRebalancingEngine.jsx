import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  AreaChart, Area, PieChart, Pie, Cell
} from 'recharts';

// ── Design Tokens (matching existing AssetOS palette) ──
const GOLD = '#d4af37';
const GOLD_DIM = '#b5952f';
const NAVY = '#0a192f';
const CHARCOAL = '#112240';
const CREAM = '#e6f1ff';
const SLATE = '#8892b0';
const ASSET_COLORS = [GOLD, '#38bdf8', SLATE, CREAM, GOLD_DIM, '#22c55e', '#8b5cf6', '#f59e0b'];

// ── Sub-components ──

const GlassCard = ({ children, className = '', delay = 0 }) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay, ease: 'easeOut' }}
    className={`glass-card p-6 ${className}`}
  >
    {children}
  </motion.div>
);

const SectionLabel = ({ children }) => (
  <h3 className="text-xl font-serif border-l-2 border-elite-gold pl-3 text-elite-cream mb-6">
    {children}
  </h3>
);

const MetricValue = ({ label, value, sub, color = 'text-elite-gold' }) => (
  <div>
    <div className="text-[10px] uppercase tracking-widest text-elite-slate mb-1">{label}</div>
    <div className={`text-xl font-mono ${color}`}>{value}</div>
    {sub && <div className="text-[10px] text-elite-slate/60 mt-0.5">{sub}</div>}
  </div>
);

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="p-3 rounded-lg border bg-elite-charcoal border-elite-gold/30 shadow-2xl">
      <p className="text-elite-gold font-serif text-xs mb-1">{label !== undefined ? `Epoch ${label}` : ''}</p>
      {payload.map((p, i) => (
        <p key={i} className="text-[11px] text-elite-cream flex justify-between gap-3">
          <span className="text-elite-slate">{p.name}:</span>
          <span className="font-mono">{Number(p.value).toFixed(3)}</span>
        </p>
      ))}
    </div>
  );
};

// ── Animated Allocation Ring (Hero Right Side) ──
const AllocationRing = ({ pieData = [] }) => {
  const [rotation, setRotation] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setRotation(r => r + 0.3), 50);
    return () => clearInterval(iv);
  }, []);

  const displayData = pieData.length > 0 ? pieData : [
    { name: 'Portfolio', value: 100 }
  ];

  return (
    <div className="relative w-64 h-64 mx-auto">
      <div className="absolute inset-0 rounded-full"
        style={{ boxShadow: `0 0 60px ${GOLD}10, 0 0 120px ${GOLD}05` }} />
      <svg viewBox="0 0 200 200" className="w-full h-full" style={{ transform: `rotate(${rotation}deg)` }}>
        {displayData.map((d, i) => {
          const total = displayData.reduce((s, x) => s + x.value, 0);
          const startAngle = displayData.slice(0, i).reduce((s, x) => s + (x.value / total) * 360, 0);
          const angle = (d.value / total) * 360;
          const r = 80, cx = 100, cy = 100;
          const rad1 = ((startAngle - 90) * Math.PI) / 180;
          const rad2 = ((startAngle + angle - 90) * Math.PI) / 180;
          const x1 = cx + r * Math.cos(rad1), y1 = cy + r * Math.sin(rad1);
          const x2 = cx + r * Math.cos(rad2), y2 = cy + r * Math.sin(rad2);
          const large = angle > 180 ? 1 : 0;
          return (
            <path key={i}
              d={`M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`}
              fill={ASSET_COLORS[i % ASSET_COLORS.length]}
              fillOpacity={0.15}
              stroke={ASSET_COLORS[i % ASSET_COLORS.length]}
              strokeWidth={1}
              strokeOpacity={0.5}
            />
          );
        })}
        <circle cx="100" cy="100" r="50" fill="none" stroke={GOLD} strokeWidth="0.5" strokeOpacity="0.3" />
        <circle cx="100" cy="100" r="30" fill="none" stroke={GOLD} strokeWidth="0.3" strokeOpacity="0.2" />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-serif text-elite-gold">PPO</div>
          <div className="text-[9px] uppercase tracking-widest text-elite-slate">Agent Active</div>
        </div>
      </div>
    </div>
  );
};

// ── Allocation Bar ──
const AllocationBar = ({ name, prev, curr, diff: rawDiff, color, delay }) => {
  const [animated, setAnimated] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), delay * 1000);
    return () => clearTimeout(t);
  }, [delay]);
  const width = animated ? curr : prev;
  // Use backend-computed diff (already rounded), fall back to client-side with rounding
  const diff = rawDiff !== undefined ? rawDiff : Math.round((curr - prev) * 10) / 10;
  return (
    <div className="flex items-center gap-4 py-3 border-b border-white/5 last:border-0">
      <div className="w-24 text-xs text-elite-slate font-sans truncate" title={name}>{name}</div>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden relative">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: `${prev}%` }}
          animate={{ width: `${width}%` }}
          transition={{ duration: 1.2, ease: 'easeInOut' }}
        />
      </div>
      <div className="w-24 text-right font-mono text-xs text-elite-cream">
        {prev}% → {curr}%
      </div>
      <div className={`w-16 text-right text-xs font-mono ${diff > 0 ? 'text-green-400' : diff < 0 ? 'text-red-400' : 'text-elite-slate'}`}>
        {diff > 0 ? '+' : ''}{diff}%
      </div>
    </div>
  );
};

// ── Main Component ──
export default function RLRebalancingEngine() {
  const [rlData, setRlData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [agentRunning, setAgentRunning] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [visibleEvents, setVisibleEvents] = useState([]);

  // Fetch RL data from backend
  const runAgent = async () => {
    setLoading(true);
    setError(null);
    setAgentRunning(true);
    try {
      const res = await fetch('/api/rl_rebalance', { method: 'POST' });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setRlData(data);
      setCurrentEpoch(data.reward_curve?.length || 60);
    } catch (err) {
      console.error("RL Engine Error:", err);
      setError(err.message);
      setAgentRunning(false);
    } finally {
      setLoading(false);
    }
  };

  // Animate epoch counter when agent running
  useEffect(() => {
    if (!agentRunning || !rlData) return;
    const iv = setInterval(() => setCurrentEpoch(e => e + 1), 3000);
    return () => clearInterval(iv);
  }, [agentRunning, rlData]);

  // Build timeline events from explain_reasons
  const timelineEvents = rlData ? [
    { time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
      event: `Regime Detected: ${rlData.regime?.name}`, type: 'alert' },
    ...(rlData.explain_reasons || []).map((r, i) => ({
      time: new Date(Date.now() + (i + 1) * 60000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
      event: r.text,
      type: i === 0 ? 'action' : i === (rlData.explain_reasons.length - 1) ? 'success' : 'action'
    })),
    { time: new Date(Date.now() + 300000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
      event: 'Portfolio Rebalanced Successfully', type: 'success' },
  ] : [];

  // Stagger timeline events
  useEffect(() => {
    if (!rlData) return;
    setVisibleEvents([]);
    const timers = timelineEvents.map((_, i) =>
      setTimeout(() => setVisibleEvents(prev => {
        if (prev.includes(i)) return prev;
        return [...prev, i];
      }), (i + 1) * 600)
    );
    return () => timers.forEach(clearTimeout);
  }, [rlData]);

  // Extract data from API response
  const metrics = rlData?.metrics || {};
  const regime = rlData?.regime || {};
  const allocData = rlData?.allocation_transition || [];
  const rewardData = rlData?.reward_curve || [];
  const explainReasons = rlData?.explain_reasons || [];
  const pieData = rlData?.pie_data || [];
  const liveData = rlData?.live_data || [];

  return (
    <div className="space-y-8 pb-20">

      {/* ─── HERO SECTION ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center py-8 border-b border-elite-gold/10">
        {/* Left */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.7 }}>
          <div className="text-[10px] uppercase tracking-[0.3em] text-elite-gold/60 mb-3">Module — Autonomous Optimization</div>
          <h1 className="text-4xl lg:text-5xl font-serif text-elite-cream leading-tight mb-4">
            Reinforcement Learning<br />
            <span className="text-elite-gold">Rebalancing Engine</span>
          </h1>
          <p className="text-elite-slate text-sm leading-relaxed max-w-lg mb-8">
            Adaptive portfolio allocation using PPO-driven autonomous optimization.
            {rlData ? ` Currently in ${regime.name} regime — ${regime.description}` : ' Click "Run Agent" to analyze your portfolio and compute optimal rebalancing.'}
          </p>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={runAgent}
              disabled={loading}
              className={`px-5 py-2.5 rounded-lg border text-sm uppercase tracking-wider transition-all duration-300
                ${agentRunning
                  ? 'border-elite-gold bg-elite-gold/10 text-elite-gold shadow-[0_0_20px_rgba(212,175,55,0.1)]'
                  : 'border-elite-gold/40 text-elite-gold/80 hover:border-elite-gold hover:bg-elite-gold/5'
                }`}
            >
              {loading ? '⟳ Computing...' : agentRunning ? '■ Agent Running' : 'Run Agent'}
            </button>
            {agentRunning && (
              <button
                onClick={() => { setAgentRunning(false); }}
                className="px-5 py-2.5 rounded-lg border border-red-500/30 text-red-400/80 hover:border-red-500 text-sm uppercase tracking-wider transition-all duration-300"
              >
                Stop Agent
              </button>
            )}
          </div>
          {error && (
            <div className="mt-4 p-3 border border-red-500/30 rounded text-red-400 text-sm">
              ⚠ {error}. Please generate a portfolio first from the Core Portfolio tab.
            </div>
          )}
        </motion.div>

        {/* Right — Allocation Ring + Mini Chart */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.7, delay: 0.2 }}
          className="flex flex-col items-center gap-6">
          <AllocationRing pieData={pieData} />
          {liveData.length > 0 && (
            <div className="w-full h-24">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={liveData}>
                  <defs>
                    <linearGradient id="rlGoldGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={GOLD} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={GOLD} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="value" stroke={GOLD} strokeWidth={1.5} fill="url(#rlGoldGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </motion.div>
      </div>

      {/* ─── CONTENT — Only shown when agent has data ─── */}
      {rlData && (
        <>
          {/* ─── ROW 1: STATUS CARDS ─── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Agent Status */}
            <GlassCard delay={0.1}>
              <SectionLabel>Agent Status</SectionLabel>
              <div className="flex items-center gap-2 mb-5">
                <span className="relative flex h-2.5 w-2.5">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${agentRunning ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${agentRunning ? 'bg-green-400' : 'bg-red-400'}`} />
                </span>
                <span className={`text-xs uppercase tracking-widest ${agentRunning ? 'text-green-400' : 'text-red-400'}`}>
                  {agentRunning ? 'Active' : 'Paused'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-y-4">
                <MetricValue label="Current Epoch" value={currentEpoch} />
                <MetricValue label="Win Rate" value={`${metrics.win_rate || 0}%`} color="text-green-400" />
                <MetricValue label="Regime" value={regime.name || 'N/A'} sub={`Vol: ${regime.avg_volatility || 0}%`} />
                <MetricValue label="Actions Taken" value={allocData.filter(a => Math.abs(a.diff) > 0.1).length} sub="This session" />
              </div>
            </GlassCard>

            {/* Current Reward */}
            <GlassCard delay={0.2}>
              <SectionLabel>Performance Metrics</SectionLabel>
              <div className="text-4xl font-mono text-elite-gold mb-1">
                {metrics.reward !== undefined ? (metrics.reward > 0 ? '+' : '') + metrics.reward : 'N/A'}
              </div>
              <div className="text-xs text-green-400 mb-5">
                Sharpe: {metrics.old_sharpe} → {metrics.sharpe}
              </div>
              <div className="grid grid-cols-2 gap-y-4">
                <MetricValue label="Sharpe Ratio" value={metrics.sharpe || 'N/A'} color="text-elite-cream" />
                <MetricValue label="Sortino" value={metrics.sortino || 'N/A'} color="text-elite-cream" />
                <MetricValue label="Max Drawdown" value={`-${metrics.max_drawdown || 0}%`} color="text-red-400" />
                <MetricValue label="Volatility" value={`${metrics.volatility || 0}%`} color="text-yellow-400" />
              </div>
            </GlassCard>

            {/* Market Regime */}
            <GlassCard delay={0.3}>
              <SectionLabel>Market Regime</SectionLabel>
              <div className="space-y-3">
                {[
                  { label: 'Bullish', color: 'green' },
                  { label: 'Normal', color: 'blue' },
                  { label: 'Volatile', color: 'yellow' },
                  { label: 'Risk-Off', color: 'red' },
                ].map(({ label, color }) => {
                  const active = regime.name === label;
                  return (
                    <div key={label}
                      className={`flex items-center justify-between px-4 py-2.5 rounded-lg border transition-all duration-300
                        ${active
                          ? 'border-elite-gold/40 bg-elite-gold/5'
                          : 'border-white/5 bg-white/[0.02]'
                        }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <div className={`w-2 h-2 rounded-full ${active ? `bg-${color}-400` : 'bg-white/10'}`}
                          style={active ? { backgroundColor: color === 'yellow' ? '#facc15' : color === 'blue' ? '#38bdf8' : color === 'green' ? '#22c55e' : '#ef4444' } : {}}
                        />
                        <span className={`text-sm ${active ? 'text-elite-cream' : 'text-elite-slate/50'}`}>{label}</span>
                      </div>
                      {active && <span className="text-[9px] uppercase tracking-widest text-elite-gold">Active</span>}
                    </div>
                  );
                })}
              </div>
            </GlassCard>
          </div>

          {/* ─── ROW 2: ALLOCATION TRANSITION ─── */}
          <GlassCard delay={0.35}>
            <div className="flex items-center justify-between mb-6">
              <SectionLabel>Allocation Transition</SectionLabel>
              <div className="text-[10px] uppercase tracking-widest text-elite-slate">PPO Epoch {currentEpoch} Rebalance</div>
            </div>
            {allocData.slice(0, 10).map((a, i) => (
              <AllocationBar
                key={a.name}
                name={a.name}
                prev={a.prev}
                curr={a.curr}
                diff={a.diff}
                color={ASSET_COLORS[i % ASSET_COLORS.length]}
                delay={0.4 + i * 0.1}
              />
            ))}
          </GlassCard>

          {/* ─── ROW 3: PPO TRAINING METRICS ─── */}
          <GlassCard delay={0.4}>
            <SectionLabel>PPO Training Metrics</SectionLabel>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                { key: 'reward', label: 'Reward Curve', color: GOLD },
                { key: 'policyLoss', label: 'Policy Loss', color: '#38bdf8' },
                { key: 'entropyLoss', label: 'Entropy Loss', color: '#8b5cf6' },
              ].map(({ key, label, color }) => (
                <div key={key}>
                  <div className="text-xs uppercase tracking-widest text-elite-slate mb-3">{label}</div>
                  <div className="h-48 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rewardData}>
                        <XAxis dataKey="epoch" stroke={SLATE} tick={{ fill: SLATE, fontSize: 9 }} tickLine={false} axisLine={false} />
                        <YAxis stroke={SLATE} tick={{ fill: SLATE, fontSize: 9 }} tickLine={false} axisLine={false} width={35} />
                        <Tooltip content={<ChartTooltip />} />
                        <Line type="monotone" dataKey={key} stroke={color} strokeWidth={1.5} dot={false} name={label} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>

          {/* ─── ROW 4: RL DECISION TIMELINE ─── */}
          <GlassCard delay={0.45}>
            <SectionLabel>RL Decision Timeline</SectionLabel>
            <div className="space-y-0.5">
              <AnimatePresence>
                {visibleEvents.map(idx => {
                  const e = timelineEvents[idx];
                  if (!e) return null;
                  const typeStyles = {
                    alert: 'text-yellow-400 border-yellow-400/20',
                    action: 'text-elite-gold border-elite-gold/20',
                    success: 'text-green-400 border-green-400/20',
                    info: 'text-elite-slate border-white/5',
                  };
                  return (
                    <motion.div key={idx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.4 }}
                      className={`flex items-center gap-4 px-4 py-3 rounded-lg border bg-white/[0.02] ${typeStyles[e.type]}`}
                    >
                      <span className="font-mono text-xs text-elite-slate/70 w-12">[{e.time}]</span>
                      <span className="w-1.5 h-1.5 rounded-full bg-current flex-shrink-0" />
                      <span className="text-sm">{e.event}</span>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </GlassCard>

          {/* ─── ROW 5: EXPLAINABILITY PANEL ─── */}
          <GlassCard delay={0.5}>
            <div className="flex items-center justify-between mb-6">
              <SectionLabel>Why AssetOS Rebalanced</SectionLabel>
              <div className="px-3 py-1 rounded-full border border-elite-gold/20 text-[9px] uppercase tracking-widest text-elite-gold/60">
                Explainable AI
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {explainReasons.map((r, i) => (
                <motion.div key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.6 + i * 0.1 }}
                  className="flex items-start gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:border-elite-gold/20 hover:bg-elite-gold/[0.03] transition-all duration-300"
                >
                  <span className="text-xl mt-0.5">{r.icon}</span>
                  <div>
                    <p className="text-sm text-elite-cream leading-relaxed">{r.text}</p>
                    <p className="text-[10px] text-elite-slate/50 mt-1 uppercase tracking-wider">Confidence: {r.confidence}%</p>
                  </div>
                </motion.div>
              ))}
            </div>
            <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between">
              <p className="text-[10px] text-elite-slate/40 uppercase tracking-widest">
                Analysis generated by AssetOS PPO Agent v2.4 — {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
              </p>
              <div className="text-[10px] text-elite-gold/40 uppercase tracking-widest font-serif italic">
                Per disciplinam, potentia.
              </div>
            </div>
          </GlassCard>
        </>
      )}

      {/* Empty state when no data */}
      {!rlData && !loading && (
        <div className="text-center py-20 text-elite-slate/50">
          <div className="text-6xl mb-6 opacity-20">⚡</div>
          <p className="text-lg font-serif text-elite-cream/30">RL Engine Awaiting Activation</p>
          <p className="text-sm mt-2">Generate a portfolio in the Core tab, then click "Run Agent" to analyze.</p>
        </div>
      )}
    </div>
  );
}
