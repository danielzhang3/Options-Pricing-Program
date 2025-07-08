import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './CSVUpload.css';

interface CSVUploadProps {
  onUploadSuccess?: (data: any) => void;
  onUploadError?: (error: string) => void;
}

const CSVUpload: React.FC<CSVUploadProps> = ({ onUploadSuccess, onUploadError }) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setError(null);
      previewCSV(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    multiple: false
  });

  const previewCSV = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n');
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        
        // Parse first 5 rows for preview
        const previewRows = lines.slice(1, 6).map(line => {
          const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
          const row: any = {};
          headers.forEach((header, index) => {
            row[header] = values[index] || '';
          });
          return row;
        });

        setPreviewData(previewRows);
      } catch (err) {
        setError('Error reading CSV file. Please check the file format.');
      }
    };
    reader.readAsText(file);
  };

  const handleUpload = async () => {
    if (!uploadedFile) return;

    setIsUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch('http://localhost:8000/api/upload-csv/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadProgress(100);
        onUploadSuccess?.(result);
        setUploadedFile(null);
        setPreviewData([]);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }
    } catch (err: any) {
      setError(err.message || 'Upload failed. Please try again.');
      onUploadError?.(err.message);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleClear = () => {
    setUploadedFile(null);
    setPreviewData([]);
    setError(null);
    setUploadProgress(0);
  };

  return (
    <div className="csv-upload">
      <h2>Upload IBKR Options Data</h2>
      <p>Upload your IBKR CSV file to import options trading data for analysis</p>

      {/* File Upload Area */}
      <div 
        {...getRootProps()} 
        className={`upload-area ${isDragActive ? 'drag-active' : ''} ${uploadedFile ? 'has-file' : ''}`}
      >
        <input {...getInputProps()} />
        {!uploadedFile ? (
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <p>Drag and drop your IBKR CSV file here, or click to browse</p>
            <p className="upload-hint">Supports .csv files only</p>
          </div>
        ) : (
          <div className="file-info">
            <div className="file-icon">📄</div>
            <div className="file-details">
              <h4>{uploadedFile.name}</h4>
              <p>Size: {(uploadedFile.size / 1024).toFixed(2)} KB</p>
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {/* Preview Section */}
      {previewData.length > 0 && (
        <div className="preview-section">
          <h3>File Preview (First 5 rows)</h3>
          <div className="preview-table">
            <table>
              <thead>
                <tr>
                  {Object.keys(previewData[0] || {}).map(header => (
                    <th key={header}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewData.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value: any, colIndex) => (
                      <td key={colIndex}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {uploadedFile && (
        <div className="upload-actions">
          <button 
            onClick={handleUpload} 
            disabled={isUploading}
            className="upload-btn"
          >
            {isUploading ? 'Uploading...' : 'Upload to Database'}
          </button>
          <button 
            onClick={handleClear} 
            disabled={isUploading}
            className="clear-btn"
          >
            Clear
          </button>
        </div>
      )}

      {/* Progress Bar */}
      {isUploading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <p>{uploadProgress}% Complete</p>
        </div>
      )}

      {/* Instructions */}
      <div className="instructions">
        <h3>Instructions</h3>
        <ul>
          <li>Export your options data from Interactive Brokers as a CSV file</li>
          <li>Make sure the file contains the required columns (symbol, trade_datetime, quantity, etc.)</li>
          <li>The system will automatically parse and validate your data</li>
          <li>Uploaded data will be stored in the database for analysis</li>
        </ul>
      </div>
    </div>
  );
};

export default CSVUpload; 