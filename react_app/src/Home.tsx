import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-800 relative">
      <div className="absolute inset-0 bg-black bg-opacity-30 z-0"></div>

      <div className="container mx-auto px-4 py-16 relative z-10 flex flex-col items-center text-center">
        <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
          Real-Time{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-pink-200">
            Emotion Detection
          </span>
        </h1>
        <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-2xl">
          Analyze facial expressions in real-time using AI-powered computer
          vision
        </p>

        <Link to="/detection">
          <button
            className="relative inline-flex items-center px-8 py-4 bg-white text-purple-600 
                        font-bold rounded-full shadow-2xl overflow-hidden transform 
                        transition-all duration-300 hover:scale-105"
          >
            <span className="relative z-10">Start Detection</span>
            <div
              className="absolute inset-0 bg-gradient-to-r from-purple-300 to-pink-300 
                         opacity-25 blur-lg"
            ></div>
          </button>
        </Link>

        <div className=" mt-20 grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
          {[
            {
              icon: (
                <svg
                  className="h-16 w-16"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M19.5 12C19.5 16.1421 16.1421 19.5 12 19.5C7.85786 19.5 4.5 16.1421 4.5 12C4.5 7.85786 7.85786 4.5 12 4.5C16.1421 4.5 19.5 7.85786 19.5 12ZM21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12ZM9.375 10.5C9.99632 10.5 10.5 9.99632 10.5 9.375C10.5 8.75368 9.99632 8.25 9.375 8.25C8.75368 8.25 8.25 8.75368 8.25 9.375C8.25 9.99632 8.75368 10.5 9.375 10.5ZM15.75 9.375C15.75 9.99632 15.2463 10.5 14.625 10.5C14.0037 10.5 13.5 9.99632 13.5 9.375C13.5 8.75368 14.0037 8.25 14.625 8.25C15.2463 8.25 15.75 8.75368 15.75 9.375ZM12 15C10.1783 15 9 13.8451 9 12.75H7.5C7.5 14.9686 9.67954 16.5 12 16.5C14.3205 16.5 16.5 14.9686 16.5 12.75H15C15 13.8451 13.8217 15 12 15Z"
                      fill="#080341"
                    ></path>{" "}
                  </g>
                </svg>
              ),
              title: "Happy Detection",
              desc: "Accurately detects joyful expressions",
            },
            {
              icon: (
                <svg
                  className="h-16 w-16"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <circle
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="#1C274C"
                      stroke-width="1.5"
                    ></circle>{" "}
                    <path
                      d="M9 17C9.85038 16.3697 10.8846 16 12 16C13.1154 16 14.1496 16.3697 15 17"
                      stroke="#1C274C"
                      stroke-width="1.5"
                      stroke-linecap="round"
                    ></path>{" "}
                    <ellipse
                      cx="15"
                      cy="10.5"
                      rx="1"
                      ry="1.5"
                      fill="#1C274C"
                    ></ellipse>{" "}
                    <ellipse
                      cx="9"
                      cy="10.5"
                      rx="1"
                      ry="1.5"
                      fill="#1C274C"
                    ></ellipse>{" "}
                  </g>
                </svg>
              ),
              title: "Sadness Recognition",
              desc: "Identifies signs of sadness and melancholy",
            },
            {
              icon: (
                <svg
                  className="h-16 w-16"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <circle
                      cx="12"
                      cy="12"
                      r="9.5"
                      stroke="#222222"
                      stroke-linecap="round"
                    ></circle>{" "}
                    <path
                      d="M8.20857 16.622C8.63044 16.2567 9.20751 15.9763 9.86133 15.7876C10.5191 15.5977 11.256 15.5 12 15.5C12.744 15.5 13.4809 15.5977 14.1387 15.7876C14.7925 15.9763 15.3696 16.2567 15.7914 16.622"
                      stroke="#222222"
                      stroke-linecap="round"
                    ></path>{" "}
                    <path
                      d="M17 8L14 10"
                      stroke="#222222"
                      stroke-linecap="round"
                    ></path>{" "}
                    <path
                      d="M7 8L10 10"
                      stroke="#222222"
                      stroke-linecap="round"
                    ></path>{" "}
                    <circle cx="8" cy="10" r="1" fill="#222222"></circle>{" "}
                    <circle cx="16" cy="10" r="1" fill="#222222"></circle>{" "}
                  </g>
                </svg>
              ),
              title: "Anger Analysis",
              desc: "Detects angry or frustrated expressions",
            },
            {
              icon: (
                <svg
                  className="h-16 w-16"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  stroke="#000000"
                  stroke-width="1"
                  stroke-linecap="round"
                  stroke-linejoin="miter"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line
                      x1="8"
                      y1="9"
                      x2="8.01"
                      y2="9"
                      stroke-width="2"
                      stroke-linecap="round"
                    ></line>
                    <line
                      x1="15.99"
                      y1="9"
                      x2="16"
                      y2="9"
                      stroke-width="2"
                      stroke-linecap="round"
                    ></line>
                    <circle cx="12" cy="15" r="3"></circle>
                  </g>
                </svg>
              ),
              title: "Surprise Detection",
              desc: "Identifies surprised or shocked looks",
            },
            {
              icon: (
                <svg
                  className="h-20 w-20"
                  fill="#000000"
                  viewBox="0 0 64 64"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <g id="scared">
                      {" "}
                      <path d="M6.7,13.4561a1,1,0,0,0-1.4141,0l-.7671.767-.767-.767A1,1,0,0,0,2.3379,14.87l.7671.7671-.7671.7671A1,1,0,1,0,3.752,17.8184l.767-.7671.7671.7671A1,1,0,1,0,6.7,16.4043l-.7671-.7671L6.7,14.87A1,1,0,0,0,6.7,13.4561Z"></path>{" "}
                      <path d="M56,53a3,3,0,1,0,3,3A3.0033,3.0033,0,0,0,56,53Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,56,57Z"></path>{" "}
                      <circle cx="59" cy="45" r="1"></circle>{" "}
                      <path d="M32,9A24,24,0,1,0,56,33,24.0275,24.0275,0,0,0,32,9Zm0,46A22,22,0,1,1,54,33,22.0248,22.0248,0,0,1,32,55Z"></path>{" "}
                      <circle cx="39" cy="25" r="2"></circle>{" "}
                      <circle cx="25" cy="25" r="2"></circle>{" "}
                      <path d="M36.8037,18.02a1,1,0,0,0,.39,1.9615c.1738-.0328,4.2841-.7876,5.9121,2.4663a1,1,0,1,0,1.789-.8946C43.043,17.8481,38.7949,17.6221,36.8037,18.02Z"></path>{" "}
                      <path d="M26.8066,19.981a1,1,0,0,0,.39-1.9615c-1.9893-.3974-6.2383-.17-8.0908,3.5332a1,1,0,0,0,1.789.8946C22.5215,19.1934,26.6328,19.9482,26.8066,19.981Z"></path>{" "}
                      <path d="M42,42a10,10,0,0,0-20,0,7.6063,7.6063,0,0,0,1.2466,4.3477c.0068.0128.0174.023.0247.0355C24.86,48.6974,27.8785,50,32,50s7.14-1.3026,8.7287-3.6168c.0073-.0125.0179-.0227.0247-.0355A7.6063,7.6063,0,0,0,42,42ZM25.4818,45.9319A3.9054,3.9054,0,0,1,28,45a3.9634,3.9634,0,0,1,3.1982,1.6245.9994.9994,0,0,0,1.6036,0A3.9634,3.9634,0,0,1,36,45a3.9054,3.9054,0,0,1,2.5182.9319C36.8042,47.6529,33.9669,48,32,48S27.1958,47.6529,25.4818,45.9319Z"></path>{" "}
                    </g>{" "}
                  </g>
                </svg>
              ),
              title: "Fear Detection",
              desc: "Identifies fearful or anxious expressions",
            },
            {
              icon: (
                <svg
                  className="h-16 w-16"
                  fill="#000000"
                  version="1.1"
                  id="Layer_1"
                  xmlns="http://www.w3.org/2000/svg"
                  xmlnsXlink="http://www.w3.org/1999/xlink"
                  viewBox="0 0 260 260"
                  xmlSpace="preserve"
                  stroke="#000000"
                >
                  <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
                  <g
                    id="SVGRepo_tracerCarrier"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  ></g>
                  <g id="SVGRepo_iconCarrier">
                    {" "}
                    <path d="M130,2C59.3,2,2,59.3,2,130s57.3,128,128,128s128-57.3,128-128S200.7,2,130,2z M71.9,133.7c-13,0-23.5-10.5-23.5-23.5 c0-2.4,0.4-4.6,1-6.8L34,102.2l1.1-14l63.4,4.9l11.8-20.4l12.1,7l-16.2,28l-11-0.8c0.2,1.1,0.2,2.2,0.2,3.3 C95.4,123.1,84.9,133.7,71.9,133.7z M190.8,196c0,0-10.3-6-37.5-2.6c-21.8,2.6-57.9,15.8-57.9,15.8s23.3-42,61.3-45.7 C194.1,159.7,190.8,196,190.8,196z M211.6,110.2c0,6.9-2.9,13-7.6,17.3c-7.6-1.7-17.2-2.3-29.4-1.4c-1.1,0.1-2.2,0.2-3.4,0.3 c-4.1-4.2-6.6-10-6.6-16.3c0-1.1,0.1-2.2,0.2-3.3l-11,0.8l-16.2-28l12.1-7l11.8,20.4l63.4-4.9l1.1,14l-15.4,1.2 C211.3,105.5,211.6,107.8,211.6,110.2z"></path>{" "}
                  </g>
                </svg>
              ),
              title: "Disgust Detection",
              desc: "Identifies disgusted or repulsed expressions",
            },
          ].map((feature, index) => (
            <div
              key={index}
              className="p-6 bg-white/10 rounded-3xl backdrop-blur-lg"
            >
              {feature.icon}
              <h3 className="text-xl font-semibold text-white mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-200">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Home;
