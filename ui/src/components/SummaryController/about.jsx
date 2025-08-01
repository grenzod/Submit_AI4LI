import React, { useState } from 'react';
import axios from 'axios';
import catGif from '../../assets/cat-running.gif';

export default function FileUpLoader() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);

  const handeFileChange = (e) => {
    setFile(e.target.files[0]);
    setSummary('');
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file to upload');
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post('http://localhost:8000/summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error uploading file: ', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      {/* File selector & button */}
      <div className="flex flex-col sm:flex-row items-center gap-4">
        <label className="w-full sm:w-2/3">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-400 transition">
            {file
              ? <span className="text-gray-700">{file.name}</span>
              : <span className="text-gray-500">Click để chọn tệp (.txt/.doc/.docx)</span>
            }
            <input
              type="file"
              accept=".txt, .doc, .docx"
              onChange={handeFileChange}
              className="hidden"
            />
          </div>
        </label>
        <button
          onClick={handleUpload}
          disabled={loading}
          className="w-full sm:w-1/3 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition"
        >
          {loading ? 'Uploading...' : 'Upload & Summarize'}
        </button>
      </div>

      {/* Summary box */}
      <div className="relative bg-white rounded-lg shadow h-64 overflow-auto">
        {loading && (
          <div className="absolute inset-0 bg-white bg-opacity-70 flex items-center justify-center rounded-lg">
            <img
              src={catGif}
              alt="Loading..."
              className="w-32 h-32 object-contain"
            />
          </div>
        )}
        <pre className="p-4 text-gray-800 whitespace-pre-wrap">
          {summary || 'Tóm tắt sẽ hiển thị ở đây...'}
        </pre>
      </div>
    </div>
  );
}
