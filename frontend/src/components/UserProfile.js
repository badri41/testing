import React from 'react';

const UserProfile = ({ userData }) => {
  return (
    <div className="user-profile">
      <div className="profile-header">
        {userData.avatar && (
          <img 
            src={userData.avatar.replace('ipfs://', 'https://ipfs.io/ipfs/')} 
            alt="Profile" 
            className="profile-avatar"
          />
        )}
        <h2>{userData.userName}</h2>
      </div>
      <div className="profile-bio">
        <p>{userData.bio}</p>
      </div>
      <div className="profile-stats">
        <div className="stat">
          <span className="stat-label">Followers</span>
          <span className="stat-value">0</span>
        </div>
        <div className="stat">
          <span className="stat-label">Total Streams</span>
          <span className="stat-value">0</span>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;