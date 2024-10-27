// backend/index.js
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import { pinFileToIPFS, pinJSONToIPFS } from '@pinata/sdk';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage,
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Security middleware
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(limiter);

// Initialize Pinata
const pinata = pinFileToIPFS(
  process.env.PINATA_API_KEY,
  process.env.PINATA_SECRET_KEY
);

// Routes
app.post('/api/upload/profile', upload.single('avatar'), async (req, res) => {
  try {
    let avatarHash = null;
    if (req.file) {
      const fileResponse = await pinata.pinFileToIPFS(req.file.buffer, {
        pinataMetadata: {
          name: `avatar-${Date.now()}`
        }
      });
      avatarHash = fileResponse.IpfsHash;
    }

    const userData = JSON.parse(req.body.userData);
    const profileData = {
      ...userData,
      avatar: avatarHash ? `ipfs://${avatarHash}` : null,
      createdAt: new Date().toISOString()
    };

    const jsonResponse = await pinata.pinJSONToIPFS(profileData, {
      pinataMetadata: {
        name: `profile-${userData.userName}`
      }
    });

    res.json({ 
      success: true, 
      ipfsHash: jsonResponse.IpfsHash 
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ 
      error: 'Failed to upload profile' 
    });
  }
});

app.get('/api/check-username/:username', async (req, res) => {
  try {
    // Implement username check logic here
    // For demo, returning false always
    res.json({ exists: false });
  } catch (error) {
    res.status(500).json({ error: 'Failed to check username' });
  }
});

app.get('/api/user/:address', async (req, res) => {
  try {
    // Implement user data fetch logic here
    // For demo, returning mock data
    res.json({
      userName: 'DemoUser',
      bio: 'Demo bio',
      avatar: null,
      walletAddress: req.params.address
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch user data' });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something broke!' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});