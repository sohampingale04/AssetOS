import { ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip, LineChart, Line } from 'recharts';
import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import MarketViews from './MarketViews';

const DARK_COLORS = ['#d4af37', '#38bdf8', '#8892b0', '#e6f1ff', '#b5952f', '#22c55e', '#8b5cf6', '#f59e0b'];
const LIGHT_COLORS = ['#B48E43', '#22543D', '#701a1a', '#0F172A', '#8B6B2F', '#166534', '#5b21b6', '#b45309'];

// INR Currency Formatter — handles Cr, L, K properly
const formatINR = (val) => {
    const num = Number(val);
    if (isNaN(num)) return '₹0';
    const abs = Math.abs(num);
    if (abs >= 10000000) return `₹${(num / 10000000).toFixed(2)}Cr`;
    if (abs >= 100000) return `₹${(num / 100000).toFixed(2)}L`;
    if (abs >= 1000) return `₹${(num / 1000).toFixed(1)}K`;
    return `₹${num.toFixed(0)}`;
};

// Short format for Y-axis (fewer decimals)
const formatINRShort = (val) => {
    const num = Number(val);
    if (isNaN(num)) return '₹0';
    const abs = Math.abs(num);
    if (abs >= 10000000) return `₹${(num / 10000000).toFixed(1)}Cr`;
    if (abs >= 100000) return `₹${(num / 100000).toFixed(1)}L`;
    if (abs >= 1000) return `₹${(num / 1000).toFixed(0)}K`;
    return `₹${num.toFixed(0)}`;
};

// Custom Tooltip for Charts
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="p-4 rounded shadow-2xl z-50 border bg-elite-charcoal border-elite-gold/30">
                <p className="text-elite-gold font-serif mb-1">{`Month ${label}`}</p>
                {payload.map((p, i) => (
                    <p key={i} className="text-xs flex justify-between gap-4 text-elite-cream">
                        <span>{p.name}:</span>
                        <span className="font-bold font-mono">{formatINR(p.value)}</span>
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

export default function Dashboard({ profile, portfolio: initialPortfolio, onReset, isDarkMode }) {
    const COLORS = isDarkMode ? DARK_COLORS : LIGHT_COLORS;
    const [portfolio, setPortfolio] = useState(initialPortfolio);
    const [isResimulating, setIsResimulating] = useState(false);
    const { allocation, risk_analysis } = portfolio;

    // Safe Access to Risk Data
    const hasRiskData = risk_analysis && risk_analysis.months;
    const horizonYears = risk_analysis?.horizon_years || risk_analysis?.simulation_info?.horizon_years || 5;
    const capital = Number(profile.capital) || 100000;

    // Process risk data for Area Chart — 5 percentile bands + sample paths
    const chartData = hasRiskData ? risk_analysis.months.map((month, i) => {
        const point = {
            month: month,
            p5: risk_analysis.p5[i],
            p25: risk_analysis.p25[i],
            median: risk_analysis.median[i],
            p75: risk_analysis.p75[i],
            p95: risk_analysis.p95[i],
        };
        // Add sample paths
        if (risk_analysis.sample_paths) {
            risk_analysis.sample_paths.forEach((path, j) => {
                point[`sample_${j}`] = path[i];
            });
        }
        return point;
    }) : [];

    // Portfolio metrics from backend (REAL computed values)
    const metrics = risk_analysis?.portfolio_metrics || {};
    const finalStats = risk_analysis?.final_stats || {};
    const simInfo = risk_analysis?.simulation_info || {};

    const scorecard = risk_analysis?.scorecard || {};
    const benchmarks = risk_analysis?.benchmarks || {};
    const goalProb = risk_analysis?.goal_probability || 0;

    // Re-simulate handler — fresh Monte Carlo
    const handleResimulate = useCallback(async () => {
        setIsResimulating(true);
        try {
            const res = await fetch('/api/resimulate', { method: 'POST' });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            setPortfolio(prev => ({
                ...prev,
                risk_analysis: data.risk_analysis
            }));
        } catch (err) {
            console.error("Resimulation failed:", err);
        } finally {
            setIsResimulating(false);
        }
    }, []);

    return (
        <div className="space-y-8 pb-20">
            {/* Header */}
            <div className="flex justify-between items-end border-b border-elite-gold/20 pb-6">
                <div>
                    <h2 className="text-3xl font-serif text-elite-gold">Executive Summary</h2>
                    <p className="text-elite-slate mt-1">Portfolio Design for <span className="text-elite-gold">{profile.name}</span></p>
                </div>
                <div className="flex flex-col items-end gap-2">
                    <button
                        onClick={onReset}
                        className="text-xs text-elite-gold hover:text-elite-goldHover underline underline-offset-4 mb-1"
                    >
                        ← NEW STRATEGY
                    </button>
                    <div className="text-right">
                        <div className="text-xs uppercase tracking-widest text-elite-slate">Total Capital</div>
                        <div className="text-2xl font-mono text-elite-gold">₹ {Number(profile.capital).toLocaleString('en-IN')}</div>
                    </div>
                </div>
            </div>

            {/* KPI Strip */}
            {hasRiskData && metrics.sharpe_ratio !== undefined && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                    className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4"
                >
                    {[
                        { label: 'Expected Return', value: `${metrics.expected_return}%`, color: 'text-green-400' },
                        { label: 'Volatility', value: `${metrics.volatility}%`, color: 'text-yellow-400' },
                        { label: 'Sharpe Ratio', value: metrics.sharpe_ratio?.toFixed(2), color: 'text-elite-gold' },
                        { label: 'Max Drawdown', value: `${metrics.max_drawdown}%`, color: 'text-red-400' },
                        { label: 'VaR (95%)', value: `${metrics.var_95}%`, color: 'text-orange-400' },
                        { label: 'Diversification', value: metrics.diversification_ratio?.toFixed(2), color: 'text-sky-400' },
                    ].map((kpi, i) => (
                        <div key={i} className={`glass-card p-4 text-center`}>
                            <div className="text-[10px] uppercase tracking-widest text-elite-slate mb-1">{kpi.label}</div>
                            <div className={`text-xl font-mono ${kpi.color}`}>{kpi.value}</div>
                        </div>
                    ))}
                </motion.div>
            )}

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Left Col: Allocation */}
                <motion.div
                    initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.2 }}
                    className="glass-card p-6 lg:col-span-1"
                >
                    <h3 className="text-xl mb-6 border-l-2 border-elite-gold pl-3 text-elite-cream">Asset Allocation</h3>
                    <div className="h-[300px] w-full relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={allocation}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                    stroke="none"
                                >
                                    {allocation.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'var(--bg-secondary)',
                                        borderColor: '#d4af37',
                                        color: 'var(--text-primary)'
                                    }}
                                    itemStyle={{ color: 'var(--text-primary)' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
                            <span className="text-3xl font-serif text-elite-cream">{allocation.length}</span>
                            <div className="text-[10px] uppercase tracking-widest text-elite-slate">Assets</div>
                        </div>
                    </div>

                    <div className="mt-4 space-y-3 max-h-[300px] overflow-y-auto pr-2">
                        {allocation.map((item, index) => (
                            <div key={index} className="flex justify-between items-center text-sm border-b border-elite-gold/10 pb-2">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                                    <span className="text-elite-slate">{item.name}</span>
                                </div>
                                <span className="font-mono text-elite-cream">{item.value}%</span>
                            </div>
                        ))}
                    </div>
                </motion.div>

                {/* Right Col: Simulation */}
                <motion.div
                    initial={{ x: 20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.4 }}
                    className="glass-card p-6 lg:col-span-2 flex flex-col gap-6"
                >
                    {/* Top Section: Area Chart */}
                    <div>
                        <div className="flex justify-between mb-6">
                            <div>
                                <h3 className="text-xl border-l-2 border-elite-gold pl-3 text-elite-cream">
                                    Projected Wealth ({horizonYears} Year{horizonYears !== 1 ? 's' : ''})
                                </h3>
                                <p className="text-xs text-elite-slate mt-1 ml-4">
                                    {simInfo.n_paths?.toLocaleString() || '1,000'} Monte Carlo paths · Regime-switching volatility
                                </p>
                            </div>
                            <div className="flex flex-col items-end gap-2">
                                <button
                                    onClick={handleResimulate}
                                    disabled={isResimulating}
                                    className="text-[10px] uppercase tracking-widest px-3 py-1.5 border border-elite-gold/30 text-elite-gold hover:bg-elite-gold/10 rounded transition-all disabled:opacity-50"
                                >
                                    {isResimulating ? '↻ Simulating...' : '↻ Re-Simulate'}
                                </button>
                                <div className="flex flex-col text-right text-xs font-mono space-y-1">
                                    <div className="text-green-400">
                                        Best Case: <span className="font-bold text-lg">{formatINR(finalStats.best_case)}</span>
                                        <span className="text-[10px] ml-1 opacity-60">({finalStats.best_case_pct > 0 ? '+' : ''}{finalStats.best_case_pct}%)</span>
                                    </div>
                                    <div className="text-elite-gold">
                                        Likely Case: <span className="font-bold text-lg">{formatINR(finalStats.median_case)}</span>
                                        <span className="text-[10px] ml-1 opacity-60">({finalStats.median_pct > 0 ? '+' : ''}{finalStats.median_pct}%)</span>
                                    </div>
                                    <div className="text-red-400">
                                        Worst Case: <span className="font-bold text-lg">{formatINR(finalStats.worst_case)}</span>
                                        <span className="text-[10px] ml-1 opacity-60">({finalStats.worst_case_pct > 0 ? '+' : ''}{finalStats.worst_case_pct}%)</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="relative h-[350px] w-full">
                            {chartData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={chartData}>
                                        <defs>
                                            <linearGradient id="colorBand95" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={isDarkMode ? "#d4af37" : "#B48E43"} stopOpacity={0.05} />
                                                <stop offset="95%" stopColor={isDarkMode ? "#d4af37" : "#B48E43"} stopOpacity={0} />
                                            </linearGradient>
                                            <linearGradient id="colorBand75" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={isDarkMode ? "#d4af37" : "#B48E43"} stopOpacity={0.1} />
                                                <stop offset="95%" stopColor={isDarkMode ? "#d4af37" : "#B48E43"} stopOpacity={0.02} />
                                            </linearGradient>
                                        </defs>
                                        <XAxis
                                            dataKey="month"
                                            stroke={isDarkMode ? "#8892b0" : "#0F172A"}
                                            tick={{ fill: isDarkMode ? "#8892b0" : "#0F172A", fontSize: 11 }}
                                            tickLine={false}
                                            axisLine={false}
                                            tickFormatter={(val) => {
                                                const yr = Math.floor(val / 12);
                                                const mo = Math.round(val % 12);
                                                if (val === 0) return 'Start';
                                                if (mo === 0 || val === chartData[chartData.length - 1]?.month) return `Y${yr}`;
                                                return '';
                                            }}
                                            interval="preserveStartEnd"
                                            label={{ value: `Investment Horizon (${horizonYears} years)`, position: 'insideBottom', offset: -5, fill: isDarkMode ? '#8892b0' : '#0F172A', fontSize: 10 }}
                                        />
                                        <YAxis
                                            stroke={isDarkMode ? "#8892b0" : "#0F172A"}
                                            tick={{ fill: isDarkMode ? "#8892b0" : "#0F172A", fontSize: 11 }}
                                            tickLine={false}
                                            axisLine={false}
                                            tickFormatter={formatINRShort}
                                            domain={['auto', 'auto']}
                                        />
                                        <Tooltip content={<CustomTooltip />} />

                                        {/* P5-P95 band (outer) */}
                                        <Area type="monotone" dataKey="p95" stroke="none" fill="url(#colorBand95)" name="P95" />
                                        <Area type="monotone" dataKey="p5" stroke="none" fill="transparent" name="P5" />

                                        {/* P25-P75 band (inner) */}
                                        <Area type="monotone" dataKey="p75" stroke="none" fill="url(#colorBand75)" name="P75" />
                                        <Area type="monotone" dataKey="p25" stroke="none" fill="transparent" name="P25" />

                                        {/* Boundary lines */}
                                        <Area type="monotone" dataKey="p95" stroke={isDarkMode ? "#22c55e" : "#22543D"} strokeWidth={1} strokeDasharray="4 4" fill="none" name="Best Case (P95)" />
                                        <Area type="monotone" dataKey="p5" stroke={isDarkMode ? "#ef4444" : "#701a1a"} strokeWidth={1} strokeDasharray="4 4" fill="none" name="Worst Case (P5)" />

                                        {/* Median line */}
                                        <Area type="monotone" dataKey="median" stroke={isDarkMode ? "#d4af37" : "#B48E43"} strokeWidth={2.5} fill="none" name="Median Path" />

                                        {/* Sample individual paths for stochastic authenticity */}
                                        {risk_analysis?.sample_paths?.map((_, j) => (
                                            <Line
                                                key={`sample_${j}`}
                                                type="monotone"
                                                dataKey={`sample_${j}`}
                                                stroke={isDarkMode ? "#8892b0" : "#64748b"}
                                                strokeWidth={0.8}
                                                strokeOpacity={0.25}
                                                dot={false}
                                                name={`Path ${j + 1}`}
                                                legendType="none"
                                            />
                                        ))}
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex items-center justify-center h-full text-elite-slate/50 italic">
                                    Risk simulation not available for 100% Cash portfolio.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Bottom Section: Scorecard, Goal Prob & Benchmarks */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 border-t border-white/5 pt-6">
                        
                        {/* Left Column: Scorecard & Goal Probability */}
                        <div className="flex flex-col gap-8">


                            {/* Goal Probability Gauge */}
                            <div>
                                <h4 className="text-sm uppercase tracking-widest text-elite-slate mb-4">Goal Probability Analysis</h4>
                                <div className="bg-black/20 border border-white/5 rounded-lg p-5 flex items-center justify-between">
                                    <div>
                                        <div className="text-xs text-elite-slate mb-1">Likelihood of reaching target wealth</div>
                                        <div className={`text-2xl font-mono ${goalProb > 80 ? 'text-green-400' : goalProb > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {goalProb}%
                                        </div>
                                    </div>
                                    <div className="w-1/2 h-2.5 bg-white/5 rounded-full overflow-hidden">
                                        <motion.div 
                                            className={`h-full ${goalProb > 80 ? 'bg-green-500' : goalProb > 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                            initial={{ width: 0 }}
                                            animate={{ width: `${goalProb}%` }}
                                            transition={{ duration: 1.5, delay: 0.4 }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Right Column: Benchmark Table */}
                        <div className="flex flex-col justify-between">
                            <div>
                                <div className="flex justify-between items-center mb-4">
                                    <h4 className="text-sm uppercase tracking-widest text-elite-slate">Strategy Comparison ({horizonYears}Y)</h4>
                                    <div className="text-[10px] uppercase text-elite-gold/50">Median Outcomes</div>
                                </div>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs text-left border-collapse">
                                        <thead>
                                            <tr className="border-b border-white/10 text-elite-slate uppercase text-[10px] tracking-wider">
                                                <th className="pb-3 font-normal">Strategy</th>
                                                <th className="pb-3 font-normal text-right">Projected Wealth</th>
                                                <th className="pb-3 font-normal text-right">Return</th>
                                                <th className="pb-3 font-normal text-right">Max DD</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5 font-mono">
                                            {/* AssetOS Row */}
                                            <tr className="bg-elite-gold/5 text-elite-gold transition-colors hover:bg-elite-gold/10">
                                                <td className="py-3.5 pl-2 font-sans font-medium flex items-center gap-2">
                                                    <div className="w-1.5 h-1.5 rounded-full bg-elite-gold animate-pulse"></div>
                                                    AssetOS Portfolio
                                                </td>
                                                <td className="py-3.5 text-right font-bold pr-2">{formatINR(finalStats.median_case)}</td>
                                                <td className="py-3.5 text-right pr-2">{finalStats.median_pct > 0 ? '+' : ''}{finalStats.median_pct}%</td>
                                                <td className="py-3.5 text-right pr-2">{metrics.max_drawdown}%</td>
                                            </tr>
                                            {/* Nifty 50 Row */}
                                            <tr className="text-elite-cream hover:bg-white/[0.02] transition-colors">
                                                <td className="py-3.5 pl-2 font-sans">Nifty 50 Proxy</td>
                                                <td className="py-3.5 text-right pr-2">{formatINR(benchmarks.nifty_50?.wealth)}</td>
                                                <td className="py-3.5 text-right pr-2 text-green-400">+{benchmarks.nifty_50?.return_pct}%</td>
                                                <td className="py-3.5 text-right pr-2 text-red-400">{benchmarks.nifty_50?.max_dd}%</td>
                                            </tr>
                                            {/* FD Row */}
                                            <tr className="text-elite-slate hover:bg-white/[0.02] transition-colors">
                                                <td className="py-3.5 pl-2 font-sans">Fixed Deposit (6.5%)</td>
                                                <td className="py-3.5 text-right pr-2">{formatINR(benchmarks.fixed_deposit?.wealth)}</td>
                                                <td className="py-3.5 text-right pr-2">+{benchmarks.fixed_deposit?.return_pct}%</td>
                                                <td className="py-3.5 text-right pr-2">{benchmarks.fixed_deposit?.max_dd}%</td>
                                            </tr>
                                            {/* Savings Row */}
                                            <tr className="text-elite-slate/60 hover:bg-white/[0.02] transition-colors">
                                                <td className="py-3.5 pl-2 font-sans">Savings Acct (3.5%)</td>
                                                <td className="py-3.5 text-right pr-2">{formatINR(benchmarks.savings?.wealth)}</td>
                                                <td className="py-3.5 text-right pr-2">+{benchmarks.savings?.return_pct}%</td>
                                                <td className="py-3.5 text-right pr-2">{benchmarks.savings?.max_dd}%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            
                            <div className="mt-6 p-4 rounded bg-white/[0.02] border border-white/5 text-[10px] text-elite-slate/60 italic leading-relaxed">
                                <span className="text-elite-gold not-italic font-semibold">Note:</span> Nifty 50 proxy assumes standard 12% expected return and 18% volatility. FD and Savings are deterministic compounded returns without volatility. AssetOS Portfolio metrics reflect the stochastic median outcome across {simInfo.n_paths?.toLocaleString() || '1,000'} Monte Carlo paths.
                            </div>
                        </div>
                    </div>
                </motion.div>

            </div>

            {/* Black-Litterman Input Section */}
            <MarketViews onOptimize={(newData) => setPortfolio(newData)} />
        </div>
    );
}
