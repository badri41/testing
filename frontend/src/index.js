// index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { Web3ReactProvider } from '@web3-react/core';
import { ethers } from 'ethers';

// Updated getLibrary function for ethers v6
function getLibrary(provider) {
  return new ethers.BrowserProvider(provider);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Web3ReactProvider getLibrary={getLibrary}>
      <App />
    </Web3ReactProvider>
  </React.StrictMode>
);