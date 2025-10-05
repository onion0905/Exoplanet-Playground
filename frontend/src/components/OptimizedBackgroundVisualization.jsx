import React, { useEffect, useRef, useState } from 'react';

const OptimizedBackgroundVisualization = () => {
  const iframeRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const containerRef = useRef(null);

  // 使用 Intersection Observer 來檢測背景是否在視窗中
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          // 一旦可見就停止觀察
          observer.disconnect();
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px' // 提前 50px 開始載入
      }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (isVisible && iframeRef.current && !isLoaded) {
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
      iframe.style.transition = 'opacity 0.8s ease-in-out';

      // 載入完成後淡入
      const handleLoad = () => {
        setIsLoaded(true);
        setTimeout(() => {
          iframe.style.opacity = '1';
        }, 100);
      };

      iframe.addEventListener('load', handleLoad);
      
      // 延遲載入，讓主要內容先渲染
      const loadTimer = setTimeout(() => {
        iframe.src = '/visualization/BackgroundExoplanet.html';
      }, 200);

      return () => {
        clearTimeout(loadTimer);
        iframe.removeEventListener('load', handleLoad);
      };
    }
  }, [isVisible, isLoaded]);

  return (
    <div ref={containerRef}>
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
      {isVisible && (
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
            transition: 'opacity 0.8s ease-in-out'
          }}
          title="Background Visualization"
          loading="lazy"
          preload="none"
        />
      )}
    </div>
  );
};

export default OptimizedBackgroundVisualization;
