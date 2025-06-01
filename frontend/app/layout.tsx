import type React from "react"
import type { Metadata } from "next"
import "./globals.css"
import Header from "@/components/header"
import { AuthProvider } from "@/contexts/auth-context"

export const metadata: Metadata = {
  title: "Colab Automation",
  description: "Explora y descubre modelos",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="es">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#F9FAFB] font-inter">
        <AuthProvider>
          <Header />
          <main className="min-h-screen pt-20">{children}</main>
        </AuthProvider>
      </body>
    </html>
  )
}
