import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';

export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);

    useEffect(() => {
        setIsTransitioning(true);
        const timer = setTimeout(() => setIsTransitioning(false), 700);
        return () => clearTimeout(timer);
    }, [collapsed]);

    return (
        <div
            className={`relative bg-gray-800 text-white h-screen flex flex-col justify-between
                border-r border-gray-700 transition-all duration-700 ease-in-out
                ${collapsed ? 'w-9' : 'w-56'}`}
        >
            <button
                onClick={() => setCollapsed(prev => !prev)}
                className="absolute top-1/2 right-0 transform -translate-y-1/2 translate-x-1/2
                bg-gray-800 p-1 rounded-full border border-gray-700 z-20
                transition-all duration-700 ease-in-out hover:bg-gray-700"
            >
                <svg
                    className={`w-7 h-7 text-white transform transition-transform duration-700 ${collapsed ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
            </button>

            <div className={`overflow-hidden transition-all duration-700 ease-in-out
                ${collapsed ? 'opacity-0 max-h-0' : 'opacity-100 max-h-screen'}`}>
                <div className="p-4 flex items-center">
                    <h1 className="text-xl font-bold flex-grow transition-opacity duration-700">ISTT</h1>
                </div>
            </div>

            <nav className={`flex-1 py-4 transition-all duration-700 ease-in-out
                ${collapsed ? 'opacity-0' : 'opacity-100'}`}>
                <ul className="space-y-1 transition-all duration-700">
                    <li>
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                `flex items-center gap-3 p-3 mx-2 hover:bg-gray-700 rounded-md transition-colors duration-700
                                ${isActive ? 'bg-gray-700' : ''}`
                            }>
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 transition-transform duration-700"
                                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                            </svg>
                            <span className="text-sm transition-all duration-700">Voice</span>
                        </NavLink>
                    </li>

                    <li>
                        <NavLink
                            to="/summary"
                            className={({ isActive }) =>
                                `flex items-center gap-3 p-3 mx-2 hover:bg-gray-700 rounded-md transition-colors duration-700
                                ${isActive ? 'bg-gray-700' : ''}`
                            }>
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 transition-transform duration-700"
                                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M7 7h10M7 11h10M7 15h6M5 3h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z" />
                            </svg>
                            <span className="text-sm transition-all duration-700">Summary</span>
                        </NavLink>
                    </li>

                    <li>
                        <NavLink
                            to="/gesture"
                            className={({ isActive }) =>
                                `flex items-center gap-3 p-3 mx-2 hover:bg-gray-700 rounded-md transition-colors duration-700
                                ${isActive ? 'bg-gray-700' : ''}`
                            }>
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 transition-transform duration-700"
                                fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M8 14V5a2 2 0 10-4 0v9m12-2V5a2 2 0 114 0v7m-8 2V3m0 13v6m-4-6v6" />
                            </svg>
                            <span className="text-sm transition-all duration-700">Gesture</span>
                        </NavLink>
                    </li>

                    <li>
                        <NavLink
                            to="/speech"
                            className={({ isActive }) =>
                                `flex items-center gap-3 p-3 mx-2 hover:bg-gray-700 rounded-md transition-colors duration-700
                                ${isActive ? 'bg-gray-700' : ''}`
                            }>
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8"                                
                                fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M7 12h10M7 15h7m-7 3h4" />
                            </svg>
                            <span className="text-sm transition-all duration-700">Speech</span>
                        </NavLink>
                    </li>
                </ul>
            </nav>

            <div className={`transition-all duration-700 ease-in-out overflow-hidden
                ${collapsed ? 'opacity-0 max-h-0' : 'opacity-100 max-h-screen'}`}>
                <div className="p-4 border-t border-gray-700">
                    <p className="text-xs text-gray-400 transition-opacity duration-700">© 2025 MyApp</p>
                </div>
            </div>
        </div>
    );
}