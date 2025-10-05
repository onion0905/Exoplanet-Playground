import React, { useEffect, useRef } from 'react';

const SimpleBackgroundVisualization = () => {
  const iframeRef = useRef(null);

  useEffect(() => {
    if (iframeRef.current) {
      const iframe = iframeRef.current;
      iframe.src = '/visualization/BackgroundExoplanet.html';
    }
  }, []);

  return (
    <div 
      style={{
        width: '100%',
        height: '100%',
        position: 'fixed',
        top: '0',
        left: '0',
        zIndex: '-1',
        background: 'black'
      }}
    >
      <iframe
        ref={iframeRef}
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          background: 'black',
          pointerEvents: 'none',
          opacity: '0',
          animation: 'fadeIn 1s ease-in-out 0.5s forwards'
        }}
        title="Background Visualization"
        loading="lazy"
      />
      
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default SimpleBackgroundVisualization;
