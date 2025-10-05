import React, { useEffect, useRef, useState } from 'react';

const BackgroundVisualization = () => {
  const iframeRef = useRef(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [shouldLoad, setShouldLoad] = useState(false);

  // 延遲載入背景，讓主要內容先載入
  useEffect(() => {
    const timer = setTimeout(() => {
      setShouldLoad(true);
    }, 100); // 100ms 後開始載入背景

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (shouldLoad && iframeRef.current) {
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
      iframe.style.transition = 'opacity 0.5s ease-in-out';

      // 載入完成後淡入
      const handleLoad = () => {
        setIsLoaded(true);
        iframe.style.opacity = '1';
      };

      iframe.addEventListener('load', handleLoad);
      iframe.src = '/visualization/BackgroundExoplanet.html';

      return () => {
        iframe.removeEventListener('load', handleLoad);
      };
    }
  }, [shouldLoad]);

  // 在背景載入前顯示黑色背景
  if (!shouldLoad) {
    return (
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
    );
  }

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
          opacity: isLoaded ? '1' : '0',
          transition: 'opacity 0.5s ease-in-out'
        }}
        title="Background Visualization"
        loading="lazy"
        preload="none"
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

export default BackgroundVisualization;
