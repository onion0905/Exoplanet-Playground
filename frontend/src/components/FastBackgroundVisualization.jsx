import React, { useEffect, useRef, useState } from 'react';

const FastBackgroundVisualization = () => {
  const iframeRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [preloadStarted, setPreloadStarted] = useState(false);

  // 預載入背景資源
  useEffect(() => {
    if (!preloadStarted) {
      setPreloadStarted(true);
      
      // 預載入 HTML 文件
      const link = document.createElement('link');
      link.rel = 'prefetch';
      link.href = '/visualization/BackgroundExoplanet.html';
      document.head.appendChild(link);

      // 預載入 Three.js 資源
      const threeLink = document.createElement('link');
      threeLink.rel = 'prefetch';
      threeLink.href = 'https://unpkg.com/three@0.160.0/build/three.module.js';
      document.head.appendChild(threeLink);

      // 短暫延遲後標記為準備就緒
      const timer = setTimeout(() => {
        setIsReady(true);
      }, 50);

      return () => {
        clearTimeout(timer);
        document.head.removeChild(link);
        document.head.removeChild(threeLink);
      };
    }
  }, [preloadStarted]);

  useEffect(() => {
    if (isReady && iframeRef.current && !isLoaded) {
      const iframe = iframeRef.current;
      
      // 設置 iframe 樣式
      iframe.style.width = '100%';
      iframe.style.height = '100%';
      iframe.style.border = 'none';
      iframe.style.background = 'black';
      iframe.style.position = 'fixed';
      iframe.style.top = '0';
      iframe.style.left = '0';
      iframe.style.zIndex = '-1';
      iframe.style.pointerEvents = 'none';
      iframe.style.opacity = '0';
      iframe.style.transition = 'opacity 0.6s ease-in-out';

      // 載入完成後淡入
      const handleLoad = () => {
        setIsLoaded(true);
        requestAnimationFrame(() => {
          iframe.style.opacity = '1';
        });
      };

      iframe.addEventListener('load', handleLoad);
      iframe.src = '/visualization/BackgroundExoplanet.html';

      return () => {
        iframe.removeEventListener('load', handleLoad);
      };
    }
  }, [isReady, isLoaded]);

  return (
    <>
      {/* 黑色背景作為 fallback */}
      <div 
        style={{
          width: '100%',
          height: '100%',
          background: 'black',
          position: 'fixed',
          top: '0',
          left: '0',
          zIndex: '-1'
        }}
      />
      
      {/* 背景 iframe */}
      {isReady && (
        <iframe
          ref={iframeRef}
          style={{
            width: '100%',
            height: '100%',
            border: 'none',
            background: 'black',
            position: 'fixed',
            top: '0',
            left: '0',
            zIndex: '-1',
            pointerEvents: 'none',
            opacity: isLoaded ? '1' : '0',
            transition: 'opacity 0.6s ease-in-out'
          }}
          title="Background Visualization"
          loading="eager"
          preload="auto"
        />
      )}
    </>
  );
};

export default FastBackgroundVisualization;
