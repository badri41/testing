// frontend/src/components/TokenBalance.js
import React from 'react';
import { useWeb3React } from '@web3-react/core';
// CommonJS require style
const { Contract, formatEther } = require('ethers');

const TokenBalance = ({ tokenAddress }) => {
  const { account, library } = useWeb3React();
  const [balance, setBalance] = React.useState('0');

  React.useEffect(() => {
    const fetchBalance = async () => {
      if (account && library && tokenAddress) {
        try {
          const contract = new Contract(
            tokenAddress,
            ['function balanceOf(address) view returns (uint256)'],
            library
          );
          const balance = await contract.balanceOf(account);
          setBalance(formatEther(balance));
        } catch (error) {
          console.error('Error fetching token balance:', error);
        }
      }
    };

    fetchBalance();
  }, [account, library, tokenAddress]);

  return (
    <div className="token-balance">
      <h3>Token Balance</h3>
      <p>{parseFloat(balance).toFixed(4)} STRM</p>
    </div>
  );
};

export default TokenBalance;