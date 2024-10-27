// frontend/src/components/StreamingControl.js
import React, { useState } from 'react';

const StreamingControl = ({ isStreaming, onStartStream, onEndStream, streamData }) => {
  const [streamTitle, setStreamTitle] = useState('');
  const [streamDescription, setStreamDescription] = useState('');

  const handleStart = async (e) => {
    e.preventDefault();
    if (!streamTitle) return;
    await onStartStream({ title: streamTitle, description: streamDescription });
    setStreamTitle('');
    setStreamDescription('');
  };

  return (
    <div className="streaming-control">
      {!isStreaming ? (
        <form onSubmit={handleStart}>
          <input
            type="text"
            placeholder="Stream Title"
            value={streamTitle}
            onChange={(e) => setStreamTitle(e.target.value)}
            required
          />
          <textarea
            placeholder="Stream Description"
            value={streamDescription}
            onChange={(e) => setStreamDescription(e.target.value)}
          />
          <button type="submit">Start Streaming</button>
        </form>
      ) : (
        <div className="active-stream">
          <h3>{streamData?.title}</h3>
          <p>{streamData?.description}</p>
          <button onClick={onEndStream} className="end-stream">
            End Stream
          </button>
        </div>
      )}
    </div>
  );
};

export default StreamingControl;