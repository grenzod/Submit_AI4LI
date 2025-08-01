import VoiceButton from './voiceButton';
import Waveform from './wareForm';
import TranscriptDisplay from './transcriptDisplay';
import useSpeech from './useSpeech';
import WaveTwoSide from './waveTwoSide';
import socketSpeech from './socketSpeech';

export default function VoiceControl() {
  const { listening, isSpeaking, conversation, volume, error, startListening, stopListening } = socketSpeech();

  return (
    <div className="w-full h-full p-4 flex flex-col items-center justify-center space-y-10">

      <div className="flex justify-center">
        <div className="flex items-center space-x-8">
          <div className="h-32 flex items-center justify-center">
            <WaveTwoSide active={listening} volume={volume} />
          </div>

          <div className="relative flex flex-col items-center">
            <Waveform active={listening} intense={isSpeaking} />
            <VoiceButton listening={listening} onStart={startListening} onStop={stopListening} />
          </div>

          <div className="h-32 flex items-center">
            <WaveTwoSide active={listening} volume={volume} />
          </div>
        </div>
      </div>

      <div className="w-full max-w-2xl">
        <TranscriptDisplay transcript={conversation} error={error} listening={listening} />
      </div>
    </div>
  );
}

