import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

export default function MarketViews({ onOptimize }) {
    const [isOpen, setIsOpen] = useState(false);
    const [assets, setAssets] = useState([]);
    const [views, setViews] = useState([]); // [{ ticker: 'RELIANCE', sentiment: 0.15 }]
    const [loading, setLoading] = useState(false);

    // Form State
    const [selectedAsset, setSelectedAsset] = useState('');
    const [sentiment, setSentiment] = useState('0.12'); // Default Neutral 12%

    useEffect(() => {
        // Fetch available assets on mount
        fetch('/api/assets')
            .then(res => res.json())
            .then(data => setAssets(data.tickers || []))
            .catch(err => console.error("Failed to load assets", err));
    }, []);

    const addView = () => {
        if (!selectedAsset) return;
        // Prevent duplicates
        if (views.find(v => v.ticker === selectedAsset)) return;

        setViews([...views, { ticker: selectedAsset, sentiment: parseFloat(sentiment) }]);
        setSelectedAsset('');
        setSentiment('0.12');
    };

    const removeView = (ticker) => {
        setViews(views.filter(v => v.ticker !== ticker));
    };

    const handleRun = async () => {
        setLoading(true);
        try {
            // Convert to format backend expects: {"TCS": 0.20}
            const viewsMap = views.reduce((acc, curr) => {
                acc[curr.ticker] = curr.sentiment;
                return acc;
            }, {});

            const res = await fetch('/api/optimize_bl', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ views: viewsMap })
            });

            const data = await res.json();
            if (data.error) throw new Error(data.error);

            onOptimize(data); // Pass parent the new portfolio
            setIsOpen(false); // Close the panel on success

        } catch (err) {
            alert(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mt-12">
            {!isOpen ? (
                <div className="text-center">
                    <button
                        onClick={() => setIsOpen(true)}
                        className="text-elite-gold border border-elite-gold px-6 py-2 rounded hover:bg-elite-gold hover:text-elite-navy transition-colors font-serif tracking-widest text-sm"
                    >
                        + INJECT MARKET VIEWS (BLACK-LITTERMAN)
                    </button>
                    <p className="text-xs text-elite-slate mt-2 italic">
                        Adjust the model based on your personal market outlook.
                    </p>
                </div>
            ) : (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    className="glass-card p-6 border border-elite-gold/20"
                >
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-xl font-serif text-elite-cream">Strategic Adjustments</h3>
                        <button onClick={() => setIsOpen(false)} className="text-elite-slate hover:text-elite-cream text-sm">Close</button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end mb-6">
                        {/* Asset Selector */}
                        <div className="col-span-1">
                            <label className="text-xs text-elite-gold uppercase tracking-widest block mb-2">Asset</label>
                            <select
                                value={selectedAsset}
                                onChange={(e) => setSelectedAsset(e.target.value)}
                                className="w-full bg-elite-charcoal border border-elite-slate/30 rounded p-2 text-elite-cream focus:border-elite-gold outline-none"
                            >
                                <option value="">Select Asset...</option>
                                {assets.map(a => (
                                    <option key={a} value={a}>{a}</option>
                                ))}
                            </select>
                        </div>

                        {/* Sentiment Selector */}
                        <div className="col-span-2">
                            <label className="text-xs text-elite-gold uppercase tracking-widest block mb-2">View / Sentiment</label>
                            <div className="flex bg-elite-charcoal rounded border border-elite-slate/30 overflow-hidden">
                                {[
                                    { label: 'Bearish (-5%)', val: '-0.05' },
                                    { label: 'Neutral (12%)', val: '0.12' },
                                    { label: 'Bullish (25%)', val: '0.25' }
                                ].map(opt => (
                                    <button
                                        key={opt.val}
                                        onClick={() => setSentiment(opt.val)}
                                        className={`flex-1 py-2 text-xs transition-colors ${sentiment === opt.val
                                            ? 'bg-elite-gold text-elite-navy font-bold'
                                            : 'text-elite-slate hover:bg-elite-cream/5'
                                            }`}
                                    >
                                        {opt.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Add Button */}
                        <div className="col-span-1">
                            <button
                                onClick={addView}
                                disabled={!selectedAsset}
                                className="w-full bg-elite-slate/20 text-elite-cream border border-elite-slate/30 py-2 rounded hover:bg-elite-gold hover:text-elite-navy hover:border-elite-gold transition-all disabled:opacity-50"
                            >
                                Add View
                            </button>
                        </div>
                    </div>

                    {/* Active Views List */}
                    {views.length > 0 && (
                        <div className="mb-6 space-y-2">
                            <p className="text-xs uppercase text-elite-slate tracking-widest mb-2">Active Views</p>
                            {views.map((v, i) => (
                                <div key={i} className="flex justify-between items-center bg-elite-cream/5 p-3 rounded border border-elite-cream/10">
                                    <span className="text-elite-cream font-mono">{v.ticker}</span>
                                    <span className={`text-sm ${v.sentiment > 0.12 ? 'text-green-400' : v.sentiment < 0 ? 'text-red-400' : 'text-elite-gold'}`}>
                                        Target: {(v.sentiment * 100).toFixed(1)}%
                                    </span>
                                    <button onClick={() => removeView(v.ticker)} className="text-elite-slate hover:text-red-400">×</button>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Run Button */}
                    {views.length > 0 && (
                        <div className="text-center pt-4 border-t border-elite-cream/10">
                            <button
                                onClick={handleRun}
                                disabled={loading}
                                className="gold-btn-filled w-full md:w-1/2 py-3 text-sm tracking-widest"
                            >
                                {loading ? 'RE-OPTIMIZING...' : 'RE-CALCULATE ALLOCATION'}
                            </button>
                        </div>
                    )}
                </motion.div>
            )}
        </div>
    );
}
