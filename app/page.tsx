'use client';

import React, { Fragment } from "react";
import Dashboard from "./components/Dashboard";

export default function Page() {
    return <Fragment>
            <Dashboard />
            <div className="absolute top-0 left-0 h-dvh shadow-2xl bg-red-200 shadow-red-500 w-20 animate-ping"></div>
            <div className="absolute top-0 right-0 h-dvh shadow-2xl bg-red-200 shadow-red-500 w-20 animate-ping"></div>

            <div className="absolute top-0 left-0 w-dvw shadow-2xl/100 bg-red-200 shadow-red-500 h-20 animate-ping"></div>
        </Fragment>
}
