
import './global.css'
import React from 'react'

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className='p-4 flex justify-center'>{children}</body>
        </html>
    )
}
