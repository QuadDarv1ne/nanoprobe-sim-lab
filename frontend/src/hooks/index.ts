/**
 * PWA & External API Hooks
 *
 * React hooks для Progressive Web App и внешних API
 */

export {
  usePWA,
  useInstallPWA,
  useOnlineStatus,
  useServiceWorker,
  usePushNotifications,
} from './usePWA';

export {
  useAPOD,
  useMarsPhotos,
  useAsteroids,
  useEarthImagery,
  useNASAImageLibrary,
  useNaturalEvents,
  useMarsRovers,
  useNASAHealth,
} from './useNASA';
