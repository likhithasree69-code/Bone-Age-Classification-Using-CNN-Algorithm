import React from 'react';
import { StyleSheet, SafeAreaView, Platform, StatusBar, ActivityIndicator, View } from 'react-native';
import { WebView } from 'react-native-webview';

// ✅ Update this IP if your WiFi IP changes (Currently 10.173.234.243)
const SERVER_URL = 'http://10.173.234.243:8000/';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#ffffff" />
      <WebView 
        source={{ uri: SERVER_URL }} 
        style={{ flex: 1 }}
        startInLoadingState={true}
        renderLoading={() => (
          <View style={styles.loading}>
            <ActivityIndicator size="large" color="#4c51bf" />
          </View>
        )}
        // WebView inside Expo needs these for file uploads (Camera/Gallery)
        originWhitelist={['*']}
        allowsInlineMediaPlayback={true}
        mediaPlaybackRequiresUserAction={false}
        javaScriptEnabled={true}
        domStorageEnabled={true}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0
  },
  loading: {
    position: 'absolute',
    height: '100%',
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff'
  }
});
