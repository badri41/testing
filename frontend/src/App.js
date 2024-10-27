import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useWeb3React } from '@web3-react/core';
import { InjectedConnector } from '@web3-react/injected-connector';
import StreamingDApp from './components/StreamingDApp';
import ProfileSetup from './components/ProfileSetup';
import LoadingSpinner from './components/common/LoadingSpinner';
import ErrorMessage from './components/common/ErrorMessage';
import './App.css';

// Initialize Web3 connector for supported networks
const injected = new InjectedConnector({
  supportedChainIds: [1, 3, 4, 5, 42, 1337] // Mainnet, testnets, and local network
});

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({ errorMessage: error.message });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          <h2>Something went wrong</h2>
          <p>{this.state.errorMessage}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-2 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// App content component with Web3 functionality
function AppContent() {
  const { account, activate, deactivate, error: web3Error } = useWeb3React();
  const [isProfileComplete, setIsProfileComplete] = useState(false);
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initial wallet connection check
  useEffect(() => {
    const connectOnLoad = async () => {
      try {
        const wasConnected = localStorage.getItem('wasConnected') === 'true';
        if (wasConnected && !account) {
          await activate(injected, undefined, true); // Added error handling parameter
        }
      } catch (err) {
        console.error('Error connecting on load:', err);
        localStorage.removeItem('wasConnected'); // Clear on error
      } finally {
        setLoading(false);
      }
    };

    connectOnLoad();
  }, [activate, account]);

  // Profile check effect
  useEffect(() => {
    const checkProfile = async () => {
      if (!account) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/user/${account}`);
        const data = await response.json();

        if (data && !data.error) {
          setUserData(data);
          setIsProfileComplete(true);
        } else {
          setIsProfileComplete(false);
        }
      } catch (err) {
        console.error('Error fetching profile:', err);
        setError('Failed to load profile data');
      } finally {
        setLoading(false);
      }
    };

    checkProfile();
  }, [account]);

  const connectWallet = async () => {
    try {
      if (window.ethereum) {
        await activate(injected, undefined, true);
        localStorage.setItem('wasConnected', 'true');
      } else {
        setError('Please install MetaMask to use this application');
      }
    } catch (err) {
      console.error('Connection error:', err);
      setError(err.message || 'Failed to connect wallet');
    }
  };

  const disconnectWallet = async () => {
    try {
      deactivate();
      localStorage.removeItem('wasConnected');
      setUserData(null);
      setIsProfileComplete(false);
    } catch (err) {
      console.error('Error disconnecting wallet:', err);
      setError('Failed to disconnect wallet');
    }
  };


  const handleProfileComplete = async (profileData) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          address: account,
          ...profileData
        })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setUserData(data);
      setIsProfileComplete(true);
    } catch (err) {
      console.error('Error saving profile:', err);
      setError('Failed to save profile data');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Error Messages */}
        {error && (
          <ErrorMessage 
            message={error} 
            onClose={() => setError(null)} 
          />
        )}
        {web3Error && (
          <ErrorMessage 
            message="Web3 connection error. Please check your wallet connection." 
            onClose={() => setError(null)} 
          />
        )}

        {/* Navigation Header */}
        <nav className="bg-white shadow-lg p-4">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-xl font-bold">Streaming DApp</h1>
            {account ? (
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-600">
                  {account.slice(0, 6)}...{account.slice(-4)}
                </span>
                <button
                  onClick={disconnectWallet}
                  className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <button
                onClick={connectWallet}
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
              >
                Connect Wallet
              </button>
            )}
          </div>
        </nav>

        {/* Main Content */}
        <main className="container mx-auto py-8">
          <Routes>
            <Route
              path="/"
              element={
                !account ? (
                  <div className="text-center">
                    <h2 className="text-2xl mb-4">Welcome to Streaming DApp</h2>
                    <p className="mb-4">Connect your wallet to get started</p>
                    <button
                      onClick={connectWallet}
                      className="bg-blue-500 text-white px-6 py-3 rounded-lg hover:bg-blue-600"
                    >
                      Connect Wallet
                    </button>
                  </div>
                ) : !isProfileComplete ? (
                  <ProfileSetup
                    account={account}
                    onProfileComplete={handleProfileComplete}
                  />
                ) : (
                  <Navigate to="/streaming" />
                )
              }
            />
            <Route
              path="/streaming"
              element={
                !account ? (
                  <Navigate to="/" />
                ) : !isProfileComplete ? (
                  <Navigate to="/" />
                ) : (
                  <StreamingDApp
                    userData={userData}
                    account={account}
                    onError={setError}
                  />
                )
              }
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

// Main App component
function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}

export default App;