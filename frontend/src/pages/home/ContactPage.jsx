import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";

function ContactPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-8 pt-32 max-w-6xl mx-auto">
        <h1 className="font-bold text-white text-5xl mb-10">
          Contact Us
        </h1>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
            For questions, feedback, or collaboration opportunities, please reach out to us.
          </p>
          <p className="mb-6">
            We're always excited to hear from fellow space enthusiasts and machine learning practitioners!
          </p>
        </div>

      </main>
    </div>
  );
}

export default ContactPage;
