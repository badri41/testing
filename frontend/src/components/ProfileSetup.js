// src/components/ProfileSetup.js
import React, { useState } from 'react';
import { uploadProfile, checkUsername } from '../services/pinataApi';

const ProfileSetup = ({ account, connectMetaMask, onProfileComplete }) => {
  const [formData, setFormData] = useState({
    userName: '',
    bio: '',
    avatar: null
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFormData(prev => ({ ...prev, avatar: file }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Check if username exists
      const exists = await checkUsername(formData.userName);
      if (exists) {
        throw new Error('Username already taken');
      }

      // Upload profile data
      const profileData = {
        ...formData,
        walletAddress: account
      };

      const ipfsHash = await uploadProfile(profileData);
      onProfileComplete({ ...profileData, ipfsHash });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="profile-setup-container">
      <h2>Complete Your Profile</h2>
      {!account ? (
        <button onClick={connectMetaMask}>Connect MetaMask</button>
      ) : (
        <form onSubmit={handleSubmit}>
          <div>
            <label>Username:</label>
            <input
              type="text"
              name="userName"
              value={formData.userName}
              onChange={handleInputChange}
              required
            />
          </div>
          <div>
            <label>Bio:</label>
            <textarea
              name="bio"
              value={formData.bio}
              onChange={handleInputChange}
              required
            />
          </div>
          <div>
            <label>Avatar:</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </div>
          {error && <div className="error">{error}</div>}
          <button type="submit" disabled={loading}>
            {loading ? 'Creating Profile...' : 'Create Profile'}
          </button>
        </form>
      )}
    </div>
  );
};

export default ProfileSetup;