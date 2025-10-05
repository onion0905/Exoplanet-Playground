import React, { Suspense, lazy } from 'react';

// 動態載入背景組件
const BackgroundVisualization = lazy(() => import('./BackgroundVisualization'));

const LazyBackgroundVisualization = () => {
  return (
    <Suspense 
      fallback={
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
      }
    >
      <BackgroundVisualization />
    </Suspense>
  );
};

export default LazyBackgroundVisualization;
