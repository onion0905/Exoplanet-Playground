import React, { useEffect, useState } from 'react';

const SimpleTrailCursor = () => {
  const [trail, setTrail] = useState([]);

  useEffect(() => {
    const updateMousePosition = (e) => {
      // 添加拖曳尾巴效果
      setTrail(prev => {
        const newTrail = [...prev, { 
          x: e.clientX, 
          y: e.clientY, 
          id: Date.now() + Math.random()
        }];
        // 只保留最近的 15 個點
        return newTrail.slice(-15);
      });
    };

    // 添加事件監聽器
    document.addEventListener('mousemove', updateMousePosition);

    // 清理舊的尾巴點
    const trailCleanup = setInterval(() => {
      setTrail(prev => prev.filter(point => Date.now() - point.id < 800));
    }, 50);

    return () => {
      document.removeEventListener('mousemove', updateMousePosition);
      clearInterval(trailCleanup);
    };
  }, []);

  return (
    <>
      {/* 拖曳尾巴效果 */}
      {trail.map((point, index) => {
        const opacity = Math.max(0, 1 - (index / trail.length) * 0.7);
        const size = Math.max(4, 20 - index * 1.2);
        
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
              transition: 'all 0.4s ease-out',
              boxShadow: `0 0 ${size * 2.5}px rgba(33, 150, 243, ${opacity * 0.8})`
            }}
          />
        );
      })}
    </>
  );
};

export default SimpleTrailCursor;
