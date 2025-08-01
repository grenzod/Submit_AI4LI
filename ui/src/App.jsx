import React from 'react';
import VoiceControl from './components/VoiceController';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/sideBar';
import FileUpLoader from './components/SummaryController/about';
import GestureRecognizer from './components/GestureRecognizer';
import SpeechController from './components/SpeechController';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 bg-gray-100 p-6 overflow-auto">
          <Routes>
            <Route path="/" element = {<VoiceControl />} />
            <Route path="/summary" element = {<FileUpLoader />} />
            <Route path="/gesture" element = {<GestureRecognizer />} />
            <Route path='/speech' element = {<SpeechController />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
