const { ethers } = require('ethers');

async function main() {
  const randomWallet = ethers.Wallet.createRandom();  // Creates a new wallet

  console.log('Address:', randomWallet.address);        // Public address
  console.log('Mnemonic:', randomWallet.mnemonic.phrase); // Backup phrase
}

main().catch(console.error);
