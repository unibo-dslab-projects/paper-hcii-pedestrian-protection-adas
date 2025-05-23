"use client";

import Link from "next/link";
import DisplaySettings from "./DisplaySettings";

function Navbar({children}: {children: React.ReactNode}) {
    return (
        <nav className="bg-white border-gray-200 dark:bg-gray-900 z-50">
            <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
                <Link href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
                    <span className="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">SimCar</span>
                </Link>
                {children}
            </div>
        </nav>
    );
}

export default Navbar;
