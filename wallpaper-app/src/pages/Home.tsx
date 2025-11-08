import {
  IonContent,
  IonHeader,
  IonPage,
  IonTitle,
  IonToolbar,
  IonSelect,
  IonSelectOption,
  IonItem,
  IonLabel,
  IonButton,
  IonLoading,
  IonGrid,
  IonRow,
  IonCol,
  IonImg,
  IonToast,
  IonActionSheet,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
} from '@ionic/react';
import { useEffect, useState } from 'react';
import APIService, { ApiConfig } from '../services/APIService';
import ImageService from '../services/ImageService';
import DynamicForm from '../components/DynamicForm';
import { Capacitor } from '@capacitor/core';
import { Network } from '@capacitor/network';
import { Filesystem } from '@capacitor/filesystem';
import './Home.css';

// Mock TodayFortune since it's not in APIService
const TodayFortune = async (): Promise<string> => {
    try {
        const response = await fetch('https://v2.xxapi.cn/api/horoscope?type=aquarius&time=today');
        const data = await response.json();
        return `${data.data.todo.yi}\n${data.data.todo.ji}`;
    } catch (error) {
        console.error('Failed to fetch fortune:', error);
        return 'Could not fetch fortune.';
    }
};


const Home: React.FC = () => {
  const [apiConfigs, setApiConfigs] = useState<ApiConfig[]>([]);
  const [selectedApiName, setSelectedApiName] = useState<string | null>(null);
  const [currentApi, setCurrentApi] = useState<ApiConfig | null>(null);
  const [payload, setPayload] = useState<Record<string, string | number | boolean | string[]>>({});
  const [loading, setLoading] = useState(false);
  const [imageUris, setImageUris] = useState<string[]>([]);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [showActionSheet, setShowActionSheet] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [fortune, setFortune] = useState<string>('');

  useEffect(() => {
    const getFortune = async () => {
      const f = await TodayFortune();
      setFortune(f);
    };
    getFortune();
    const loadApis = async () => {
      const configs = await APIService.loadApiConfigs();
      setApiConfigs(configs);
      if (configs.length > 0) {
        setSelectedApiName(configs[0].friendly_name);
        setCurrentApi(configs[0]);
      }
    };
    loadApis();
  }, []);

  useEffect(() => {
    if (selectedApiName) {
      const api = apiConfigs.find(api => api.friendly_name === selectedApiName);
      setCurrentApi(api || null);
    }
  }, [selectedApiName, apiConfigs]);

  const handleRequest = async () => {
    const networkStatus = await Network.getStatus();
    if (!networkStatus.connected) {
      setToastMessage('No internet connection');
      setShowToast(true);
      return;
    }

    if (!currentApi) return;
    setLoading(true);
    setImageUris([]);
    try {
      const result = await APIService.requestApi(currentApi, payload);
      const parsedResult = APIService.parseResponse(result, currentApi.response.image.path) as string | string[];
      const uris = await ImageService.saveImage(parsedResult, currentApi);
      setImageUris(uris);
    } catch (error) {
      console.error('API request failed:', error);
      setToastMessage('Failed to generate wallpaper');
      setShowToast(true);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (uri: string) => {
    setSelectedImage(uri);
    setShowActionSheet(true);
  };

  const confirmDelete = async () => {
    if (selectedImage) {
      try {
        await Filesystem.deleteFile({ path: selectedImage });
        setImageUris(imageUris.filter((uri) => uri !== selectedImage));
      } catch (error) {
        console.error('Failed to delete image:', error);
        setToastMessage('Failed to delete image');
        setShowToast(true);
      }
    }
    setShowActionSheet(false);
    setSelectedImage(null);
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Wallpaper Generator</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonItem>
          <IonLabel>Select API</IonLabel>
          <IonSelect
            value={selectedApiName}
            placeholder="Select One"
            onIonChange={(e) => setSelectedApiName(e.detail.value)}
          >
            {apiConfigs.map((api) => (
              <IonSelectOption key={api.friendly_name} value={api.friendly_name}>
                {api.friendly_name}
              </IonSelectOption>
            ))}
          </IonSelect>
        </IonItem>

        {fortune && (
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>Today's Fortune</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              {fortune.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </IonCardContent>
          </IonCard>
        )}
        
        <DynamicForm apiConfig={currentApi} onFormChange={setPayload} />

        <IonButton expand="full" onClick={handleRequest} disabled={loading}>
          Generate
        </IonButton>

        <IonLoading isOpen={loading} message={'Generating...'} />

        <IonToast
          isOpen={showToast}
          onDidDismiss={() => setShowToast(false)}
          message={toastMessage}
          duration={2000}
        />

        <IonGrid>
          <IonRow>
            {imageUris.map((uri) => (
              <IonCol size="6" key={uri}>
                <IonImg src={Capacitor.convertFileSrc(uri)} />
                <IonButton color="danger" expand="full" onClick={() => handleDelete(uri)}>
                  Delete
                </IonButton>
              </IonCol>
            ))}
          </IonRow>
        </IonGrid>

        <IonActionSheet
          isOpen={showActionSheet}
          onDidDismiss={() => setShowActionSheet(false)}
          header="Are you sure you want to delete this image?"
          buttons={[
            {
              text: 'Delete',
              role: 'destructive',
              handler: confirmDelete,
            },
            {
              text: 'Cancel',
              role: 'cancel',
            },
          ]}
        />
      </IonContent>
    </IonPage>
  );
};

export default Home;
