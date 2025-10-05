import React, { useEffect, useRef } from 'react';

const CSSBackgroundVisualization = () => {
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
          transition: 'opacity 0.8s ease-in-out'
        }}
        title="Background Visualization"
        loading="lazy"
        onLoad={(e) => {
          // 載入完成後淡入
          setTimeout(() => {
            e.target.style.opacity = '1';
          }, 100);
        }}
      />
      {/* 半透明黑色遮罩 */}
      <div 
        style={{
          position: 'absolute',
          top: '0',
          left: '0',
          width: '100%',
          height: '100%',
          background: 'rgba(0, 0, 0, 0.4)',
          pointerEvents: 'none',
          zIndex: '1'
        }}
      />
    </div>
  );
};

export default CSSBackgroundVisualization;
