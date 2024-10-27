// frontend/src/components/ViewersList.js
import React from 'react';

const ViewersList = ({ viewers }) => {
  return (
    <div className="viewers-list">
      <h3>Current Viewers ({viewers.length})</h3>
      <div className="viewers">
        {viewers.map(viewer => (
          <div key={viewer.id} className="viewer">
            <span className="viewer-name">{viewer.name}</span>
            {viewer.isSubscriber && (
              <span className="subscriber-badge">⭐</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ViewersList;