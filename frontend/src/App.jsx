import { useState } from 'react';
import Onboarding from './components/Onboarding';
import Dashboard from './components/Dashboard';
import RLRebalancingEngine from './components/RLRebalancingEngine';
import { AnimatePresence, motion } from 'framer-motion';

import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    console.error("Dashboard Crash:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-10 text-red-500 bg-white">
          <h1 className="text-2xl font-bold">Something went wrong.</h1>
          <pre className="mt-4 text-xs bg-gray-100 p-4 overflow-auto">
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {

  const [profile, setProfile] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('core');

  // Toggle Theme Class on Body
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    if (isDarkMode) {
      document.body.classList.add('light-mode');
    } else {
      document.body.classList.remove('light-mode');
    }
  };

  return (
    <div className={`min-h-screen flex flex-col transition-colors duration-300 ${isDarkMode ? 'bg-elite-navy text-elite-slate' : 'bg-[#fcfcfc] text-[#4a5568]'} selection:bg-elite-gold selection:text-elite-navy overflow-x-hidden`}>
      <nav className={`w-full px-8 py-4 flex justify-between items-center border-b ${isDarkMode ? 'border-white/5 bg-elite-navy/90' : 'border-elite-gold/20 bg-white/90'} backdrop-blur-sm fixed top-0 z-50 transition-all duration-300`}>
        <div className="flex items-center gap-4">
          <div className="text-2xl font-serif text-elite-gold tracking-widest cursor-pointer" onClick={() => { setProfile(null); setPortfolio(null); setActiveTab('core'); }}>ASSET OS</div>
          <button onClick={toggleTheme} className="p-2 rounded-full hover:bg-elite-gold/10 transition-colors" title="Toggle Theme">
            {isDarkMode ? '☀️' : '🌙'}
          </button>
        </div>

        {/* Tab Navigation — only visible when dashboard is active */}
        {profile && portfolio && (
          <div className={`flex gap-1 p-1 rounded-lg border ${isDarkMode ? 'border-white/10 bg-black/20' : 'border-black/10 bg-white/50'}`}>
            <button
              id="tab-core"
              onClick={() => setActiveTab('core')}
              className={`px-5 py-1.5 rounded-md text-xs uppercase tracking-widest transition-all duration-300 font-sans ${activeTab === 'core' ? (isDarkMode ? 'bg-elite-gold/20 text-elite-gold border border-elite-gold/30' : 'bg-[#B48E43]/20 text-[#B48E43] border border-[#B48E43]/30') : (isDarkMode ? 'text-elite-slate hover:text-elite-cream border border-transparent' : 'text-[#4a5568] hover:text-[#1a202c] border border-transparent')}`}
            >
              Core Portfolio
            </button>
            <button
              id="tab-rl"
              onClick={() => setActiveTab('rl')}
              className={`px-5 py-1.5 rounded-md text-xs uppercase tracking-widest transition-all duration-300 font-sans ${activeTab === 'rl' ? (isDarkMode ? 'bg-elite-gold/20 text-elite-gold border border-elite-gold/30' : 'bg-[#B48E43]/20 text-[#B48E43] border border-[#B48E43]/30') : (isDarkMode ? 'text-elite-slate hover:text-elite-cream border border-transparent' : 'text-[#4a5568] hover:text-[#1a202c] border border-transparent')}`}
            >
              RL Engine
            </button>
          </div>
        )}

        <div className="text-xs uppercase tracking-[0.2em] opacity-50">Private Wealth Engine</div>
      </nav>

      <main className="flex-grow pt-24 px-4 container mx-auto mb-10">
        <AnimatePresence mode='wait'>
          {!profile && (
            <motion.div
              key="onboarding"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <Onboarding setProfile={setProfile} setPortfolio={setPortfolio} isDarkMode={isDarkMode} />
            </motion.div>
          )}



          {profile && portfolio && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8 }}
            >
              <ErrorBoundary>
                {activeTab === 'core' ? (
                  <Dashboard profile={profile} portfolio={portfolio} onReset={() => { setProfile(null); setPortfolio(null); setActiveTab('core'); }} isDarkMode={isDarkMode} />
                ) : (
                  <RLRebalancingEngine />
                )}
              </ErrorBoundary>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className={`w-full py-6 text-center border-t ${isDarkMode ? 'border-white/5 bg-elite-navy' : 'border-elite-gold/10 bg-[#f8f9fa]'} mt-auto transition-colors duration-300`}>
        <p className="text-elite-gold/30 font-serif italic tracking-wider text-sm">
          "Per disciplinam, potentia."
        </p>
        <p className="text-[10px] opacity-30 mt-2 uppercase tracking-widest">
          AssetOS Elite &copy; {new Date().getFullYear()}
        </p>
      </footer>
    </div>
  );
}

export default App;
