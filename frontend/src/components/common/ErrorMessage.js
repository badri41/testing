// src/components/common/ErrorMessage.js
const ErrorMessage = ({ message, onClose }) => (
    <div className="error-message">
      <p>{message}</p>
      {onClose && <button onClick={onClose}>✕</button>}
    </div>
  );

export default ErrorMessage;