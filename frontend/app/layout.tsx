import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Header from "@/components/header"
import { AuthProvider } from "@/contexts/auth-context"

const inter = Inter({ subsets: ["latin"] })

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
      <body className={`${inter.className} bg-[#F9FAFB]`}>
        <AuthProvider>
          <Header />
          <main className="min-h-screen pt-20">{children}</main>
        </AuthProvider>
      </body>
    </html>
  )
}
