import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Button } from '@mui/material';

function LearnPage() {
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
        <h1 className="font-bold text-white text-5xl mb-10 mx-auto">
          Learn About Exoplanets
        </h1>

        <h2 className="font-bold text-white text-2xl mb-10 mx-auto">
          Brief introduction of exoplanets
        </h2>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
            An exoplanet is any planet that orbits a star outside our solar system. Some even drift freely through space without a host star — these are called rogue planets. NASA has confirmed over 6,000 exoplanets, with billions more likely waiting to be discovered.
          </p>
          <p className="mb-6">
            Below is a video by NASA, introducing what an exoplanet is.
          </p>
          <iframe width="560" height="315" src="https://www.youtube.com/embed/0ZOhJe_7GrE?si=EYmh1lZ1uJV3UyGl" title="What Is an Exoplanet?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        </div>

        <h2 className="font-bold text-white text-2xl mb-10 mx-auto">
          Types of exoplanets
        </h2>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
          Exoplanets come in diverse types, each with unique compositions and appearances:
          </p>
          <ul className="mb-6 list-disc list-inside ml-5">
            <li>Gas Giants — Planets similar to or larger than Jupiter or Saturn, composed mostly of hydrogen and helium. Hot Jupiters orbit close to their stars and reach extreme temperatures.</li>
            <li>Neptunian Planets — Similar in size to Neptune or Uranus, with mixed interiors and thick hydrogen-helium atmospheres. Mini-Neptunes are smaller versions, between Earth and Neptune in size.</li>
            <li>Super-Earths — Rocky planets more massive than Earth but lighter than Neptune; they may or may not have atmospheres.</li>
            <li>Terrestrial Planets — Earth-sized or smaller, made of rock, silicate, water, or carbon. Some may have atmospheres or oceans, hinting at potential habitability.</li>
          </ul>
          <img src="/exo_type.png" alt="Types of exoplanets" className="w-full h-auto m-10" />
          <p className="mb-6">
          Reference: <a href="https://science.nasa.gov/exoplanets/planet-types/" className="text-blue-500">https://science.nasa.gov/exoplanets/planet-types/</a>
          </p>
        </div>

        <h2 className="font-bold text-white text-2xl mb-10 mx-auto">
          Identification Methods
        </h2>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
          Scientists use several techniques to discover exoplanets, with the transit and radial velocity methods being the most common.
          </p>
          <ul className="mb-6 list-disc list-inside ml-5">
            <li>Transit Method — When a planet passes in front of its host star, it temporarily blocks a small portion of the starlight. This tiny dip in brightness allows astronomers to infer the presence, size, and orbit of the exoplanet.</li>
            <li>Radial Velocity Method — As a planet orbits, its gravity causes the star to wobble slightly. This motion shifts the color of the star’s light: toward blue when it moves closer and toward red when it moves away. By measuring these subtle shifts in the star’s spectrum, scientists can detect the unseen planet’s presence and estimate its mass.</li>
          </ul>
        </div>
        <h2 className="font-bold text-white text-2xl mb-10 mx-auto">
          Our methods - Machine Learning (AI/ML) Methods
        </h2>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
          Exoplanet identification has become an increasingly active area of astronomical research. Several major survey missions — Kepler, K2, and TESS — have been launched with the goal of detecting exoplanets using the transit method, which measures slight dips in a star’s brightness when a planet passes in front of it.
          </p>
          <p className="mb-6">
          Kepler operated for nearly a decade, providing a vast amount of data later expanded by its successor, K2. Since 2018, TESS (Transiting Exoplanet Survey Satellite) has continued this work, surveying almost the entire sky. Each mission’s data — including confirmed planets, candidates, and false positives — is now publicly available, containing key features such as orbital period, transit duration, and planetary radius.
          </p>
          <p className="mb-6">
          With these rich datasets, researchers have increasingly turned to machine learning (ML) to automate exoplanet identification. Studies show that ML models, when paired with effective data preprocessing, can achieve high accuracy in distinguishing true planets from false detections. By leveraging open NASA datasets, AI and ML techniques have the potential to uncover previously unnoticed exoplanets hidden within the data.
          </p>
        </div>

        <h2 className="font-bold text-white text-2xl mb-10 mx-auto">
        Confirmed vs. Candidate
        </h2>
        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
          An exoplanet candidate is a potential planet detected by a telescope, but not yet proven to exist. Some of these candidates later turn out to be false positives — signals that mimic planets but are caused by other factors, such as stellar activity or instrument noise.
          </p>
          <p className="mb-6">
          A planet becomes confirmed only after follow-up observations, typically by two independent telescopes, verify its existence. Thousands of candidates are still awaiting confirmation, but telescope time and computational resources are limited.
          </p>
          <p className="mb-6">
          This is where citizen scientists can contribute: by analyzing NASA’s open exoplanet data, volunteers can help identify subtle brightness dips — signals that automated systems might miss — and assist in discovering new worlds beyond our solar system.
          </p>
          <p className="mb-6">
          Reference: <a href="https://science.nasa.gov/exoplanets/planet-types/" className="text-blue-500">https://science.nasa.gov/exoplanets/planet-types/</a>
          </p>
        </div>
      </main>
    </div>
  );
}

export default LearnPage;
