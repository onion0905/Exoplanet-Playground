import React, { useEffect, useState } from 'react';

const AdvancedCustomCursor = () => {
  const [trail, setTrail] = useState([]);

  useEffect(() => {
    const updateMousePosition = (e) => {
      // 添加拖曳尾巴效果
      setTrail(prev => {
        const newTrail = [...prev, { 
          x: e.clientX, 
          y: e.clientY, 
          id: Date.now() + Math.random(),
          opacity: 1
        }];
        // 只保留最近的 12 個點
        return newTrail.slice(-12);
      });
    };

    // 添加事件監聽器
    document.addEventListener('mousemove', updateMousePosition);

    // 清理舊的尾巴點
    const trailCleanup = setInterval(() => {
      setTrail(prev => prev.filter(point => Date.now() - point.id < 600));
    }, 50);

    return () => {
      document.removeEventListener('mousemove', updateMousePosition);
      clearInterval(trailCleanup);
    };
  }, []);

  // 動態生成 CSS
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes trailFade {
        0% {
          opacity: 1;
          transform: scale(1);
        }
        100% {
          opacity: 0;
          transform: scale(0.2);
        }
      }
    `;
    document.head.appendChild(style);

    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <>
      {/* 拖曳尾巴效果 */}
      {trail.map((point, index) => {
        const opacity = Math.max(0, 1 - (index / trail.length) * 0.9);
        const size = Math.max(3, 16 - index * 1.1);
        
        return (
          <div
            key={point.id}
            style={{
              position: 'fixed',
              left: point.x - size / 2,
              top: point.y - size / 2,
              width: `${size}px`,
              height: `${size}px`,
              background: `radial-gradient(circle, 
                rgba(33, 150, 243, ${opacity * 0.9}) 0%, 
                rgba(156, 39, 176, ${opacity * 0.7}) 30%,
                rgba(255, 193, 7, ${opacity * 0.5}) 60%,
                rgba(33, 150, 243, 0) 100%
              )`,
              borderRadius: '50%',
              pointerEvents: 'none',
              zIndex: 9999,
              animation: 'trailFade 0.6s ease-out forwards',
              boxShadow: `0 0 ${size * 1.5}px rgba(33, 150, 243, ${opacity * 0.7})`
            }}
          />
        );
      })}
    </>
  );
};

export default AdvancedCustomCursor;
