import React, { useState } from 'react'; // Removed unused useEffect
import { useWeb3React } from '@web3-react/core';
// Removed unused ethers import

// Import all sub-components
import ChatBox from './ChatBox';
import StreamingControl from './StreamingControl';
import TokenBalance from './TokenBalance';
import UserProfile from './UserProfile';
import ViewersList from './ViewersList';

const StreamingDApp = () => {
  // State management
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamData, setStreamData] = useState(null);
  const [viewers, setViewers] = useState([]);
  const { account } = useWeb3React();

  // Mock user data - replace with actual user data fetching
  const userData = {
    userName: "Demo User",
    bio: "Welcome to my stream!",
    avatar: "ipfs://QmExample...",
    address: account // Using account here to connect it with user data
  };

  // Mock token address - replace with actual token contract address
  const tokenAddress = "0x1234567890123456789012345678901234567890";

  const handleStartStream = async (streamInfo) => {
    try {
      // Add your stream start logic here
      setStreamData({
        ...streamInfo,
        streamerAddress: account // Including account in stream data
      });
      setIsStreaming(true);
      // Mock viewers for demo
      setViewers([
        { id: 1, name: "Viewer1", isSubscriber: true },
        { id: 2, name: "Viewer2", isSubscriber: false }
      ]);
    } catch (error) {
      console.error("Error starting stream:", error);
    }
  };

  const handleEndStream = async () => {
    try {
      // Add your stream end logic here
      setStreamData(null);
      setIsStreaming(false);
      setViewers([]);
    } catch (error) {
      console.error("Error ending stream:", error);
    }
  };

  return (
    <div className="streaming-dapp">
      <div className="left-sidebar">
        <UserProfile userData={userData} />
        <TokenBalance tokenAddress={tokenAddress} userAddress={account} />
      </div>

      <div className="main-content">
        <StreamingControl
          isStreaming={isStreaming}
          onStartStream={handleStartStream}
          onEndStream={handleEndStream}
          streamData={streamData}
          streamerAddress={account}
        />
        
        {isStreaming && (
          <div className="stream-container">
            {/* Add your video streaming component here */}
            <div className="video-placeholder">
              Stream Content Goes Here
            </div>
          </div>
        )}
      </div>

      <div className="right-sidebar">
        <ViewersList viewers={viewers} />
        <ChatBox 
          streamId={streamData?.id} 
          userAddress={account}
        />
      </div>

      <style jsx>{`
        .streaming-dapp {
          display: grid;
          grid-template-columns: 250px 1fr 300px;
          gap: 20px;
          padding: 20px;
          height: 100vh;
        }

        .left-sidebar,
        .right-sidebar {
          background: #f5f5f5;
          padding: 16px;
          border-radius: 8px;
          height: calc(100vh - 40px);
          overflow-y: auto;
        }

        .main-content {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .stream-container {
          flex: 1;
          background: #000;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .video-placeholder {
          color: white;
          font-size: 24px;
        }
      `}</style>
    </div>
  );
};

export default StreamingDApp;