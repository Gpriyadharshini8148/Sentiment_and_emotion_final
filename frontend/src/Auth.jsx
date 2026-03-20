import { useState } from 'react'
import { supabase } from './supabaseClient'
import { motion, AnimatePresence } from 'framer-motion'
import { LogIn, UserPlus, LogOut, Mail, Lock, Loader2 } from 'lucide-react'

export default function Auth({ session, setSession }) {
  const [loading, setLoading] = useState(false)
  const [isSignUp, setIsSignUp] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState(null)

  const handleAuth = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage(null)

    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setMessage({ type: 'success', text: 'Check your email for the confirmation link!' })
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
      }
    } catch (error) {
      setMessage({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    setSession(null)
  }

  if (session) {
    return (
      <div className="auth-status glass-panel">
        <div className="user-info">
          <div className="avatar">
            {session.user.email[0].toUpperCase()}
          </div>
          <span>{session.user.email}</span>
        </div>
        <button onClick={handleLogout} className="logout-btn">
          <LogOut size={16} /> Logout
        </button>
      </div>
    )
  }

  return (
    <motion.div 
      className="auth-container glass-card"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div className="auth-header">
        <h2>{isSignUp ? 'Create Account' : 'Welcome Back'}</h2>
        <p>{isSignUp ? 'Join Tanglish AI today' : 'Sign in to your account'}</p>
      </div>

      <form onSubmit={handleAuth} className="auth-form">
        <div className="input-field">
          <Mail size={18} />
          <input
            type="email"
            placeholder="Email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="input-field">
          <Lock size={18} />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        <button type="submit" className="auth-submit" disabled={loading}>
          {loading ? <Loader2 className="animate-spin" /> : (isSignUp ? <UserPlus size={18} /> : <LogIn size={18} />)}
          {isSignUp ? 'Sign Up' : 'Sign In'}
        </button>
      </form>

      <AnimatePresence>
        {message && (
          <motion.div 
            className={`auth-message ${message.type}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            {message.text}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="auth-toggle">
        {isSignUp ? 'Already have an account?' : "Don't have an account?"}
        <button onClick={() => setIsSignUp(!isSignUp)}>
          {isSignUp ? 'Sign In' : 'Sign Up'}
        </button>
      </div>
    </motion.div>
  )
}
