import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from '@mui/material';

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const navigationLinks = [
    { label: "About us", path: "/about" }, 
    { label: "Contact", path: "/contact" }
  ];

  return (
    <header className="fixed top-0 left-0 w-full h-20 flex items-center justify-between px-[2.875rem] bg-[#39415399] shadow-[0_0.25rem_0.25rem_#00000040] z-50">
      <img
        className="h-12 object-cover cursor-pointer"
        alt="International Space Apps Challenge Logo"
        src="/logo.png"
        onClick={() => navigate("/")}
      />

      <nav className="flex gap-10">
        {navigationLinks.map((link, index) => (
          <Button
            key={index}
            variant="text"
            onClick={() => navigate(link.path)}
            sx={{
              color: 'white',
              fontSize: '1rem',
              fontWeight: 400,
              textTransform: 'none',
              '&:hover': {
                backgroundColor: 'transparent',
                opacity: 0.8
              }
            }}
          >
            {link.label}
          </Button>
        ))}
      </nav>
    </header>
  );
}

export default Navbar;
