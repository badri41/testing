// src/services/pinataApi.js
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const api = axios.create({
  baseURL: BACKEND_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadProfile = async (profileData) => {
  try {
    const formData = new FormData();
    
    // Add avatar file if it exists
    if (profileData.avatar) {
      formData.append('avatar', profileData.avatar);
    }
    
    // Add other profile data
    formData.append('userData', JSON.stringify({
      userName: profileData.userName,
      bio: profileData.bio,
      walletAddress: profileData.walletAddress
    }));

    const response = await api.post('/upload/profile', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data.ipfsHash;
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Failed to upload profile');
  }
};

export const checkUsername = async (username) => {
  try {
    const response = await api.get(`/check-username/${username}`);
    return response.data.exists;
  } catch (error) {
    throw new Error('Failed to check username availability');
  }
};

export const fetchUserData = async (address) => {
  try {
    const response = await api.get(`/user/${address}`);
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch user data');
  }
};
