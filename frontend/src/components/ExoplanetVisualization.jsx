import React, { useEffect, useRef } from 'react';

const ExoplanetVisualization = ({ width = 800, height = 400 }) => {
  const iframeRef = useRef(null);

  useEffect(() => {
    if (iframeRef.current) {
      // 創建 iframe 來載入原始的 HTML 文件
      const iframe = iframeRef.current;
      iframe.src = '/visualization/Exoplanet.html';
      iframe.style.width = `${width}px`;
      iframe.style.height = `${height}px`;
      iframe.style.border = 'none';
      iframe.style.background = 'black';
    }
  }, [width, height]);

  return (
    <iframe
      ref={iframeRef}
      style={{
        width: '100%',
        height: '100%',
        border: 'none',
        background: 'black'
      }}
      title="Exoplanet Visualization"
    />
  );
};

export default ExoplanetVisualization;
