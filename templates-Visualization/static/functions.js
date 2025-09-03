import * as THREE from "three";
//import { VertexNormalsHelper } from 'three/addons/helpers/VertexNormalsHelper.js';

// Connections based on the SMPLX model with 24 joints
export const connections = [
  [0, 1], [0, 2], [0, 3], [1, 4],
  [2, 5], [3, 6], [4, 7], [5, 8],
  [6, 9], [7, 10], [8, 11], [9, 12],
  [9, 13], [9, 14], [13, 16],
  [14, 17], [16, 18], [17, 19], [18, 20], 
  [19, 21], [12, 22],
];
// Connections based on the HMR2.0 model with 25 joints
export const connectionsPT = [
  [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22,23],
  [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20],
  [0, 15], [0, 16], [15, 17], [16, 18]
];

export const connectionsColors = [
  0xffff00, 0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0x00ffff, 0xff0000,
  0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff, 0xff0000, 0x00ff00,
  0x0000ff, 0xff00ff, 0xffff00, 0x00ffff, 0xff0000, 0x00ff00, 0x0000ff,
];

export const idColors = [
  0xffffff, 0xdd22dd, 0xdd2222, 0x22dd22, 0x2222dd, 0xdddd22, 0x22dddd,
  0x224422, 0x222244, 0x442244, 0x444422, 0x224444, 0xaa2222, 0x22aa22,
  0x882222, 0x228822, 0x222288, 0x882288, 0x888822, 0x228888, 0x442222,
  0xffffff, 0xff2222, 0x22ff22, 0x2222ff, 0xff22ff, 0xffff22, 0x22ffff,
  0x2222aa, 0xaa2288, 0xaaaa22, 0x22aaaa,
];

export let checkerboardGroup = null; 
export let checkerboardMaterial1 = null;
export let checkerboardMaterial2 = null;
let checkerboardShadowMaterial = null;

export function generateTrackList(allTracks, allTracksGroup) {
  const trackListContainer = document.getElementById("trackList");
  trackListContainer.innerHTML = "";
  allTracks.forEach((track, index) => {
    if (track.length > 1) {
      const startFrame = track[0][0];
      const trackSize = track.length;
      const listItem = document.createElement("li");
      listItem.style.marginBottom = "10px";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.style.marginRight = "10px";
      checkbox.addEventListener("change", () => {
        allTracksGroup.children[index].visible = checkbox.checked;
      });
      const label = document.createElement("label");
      label.textContent = `Track ${index + 1} (Frame: ${startFrame}, Taille: ${trackSize})`;
      listItem.appendChild(checkbox);
      listItem.appendChild(label);
      trackListContainer.appendChild(listItem);
    }
  });
}

export function initTHREE(renderer, scene, camera, controls) {
  camera.position.y = 2;
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x606060);
  renderer.sortObjects = true;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.xr.enabled = true;
  document.body.appendChild(renderer.domElement);
  controls.target.set(0, 0, -5);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.maxDistance = 20;
  controls.minDistance = 1;
  const ambientLight = new THREE.AmbientLight(0xcccccc);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.95);
  directionalLight.position.set(-3, 7, 3);
  directionalLight.castShadow = true;
  directionalLight.shadow.camera.near = 0.5;
  directionalLight.shadow.camera.far = 20;
  directionalLight.shadow.camera.top = 9;
  directionalLight.shadow.camera.bottom = -4;
  directionalLight.shadow.camera.left = -12;
  directionalLight.shadow.camera.right = 5;
  directionalLight.shadow.mapSize.width = 4096;
  directionalLight.shadow.mapSize.height = 4096;
  scene.add(directionalLight);
  renderer.setSize(window.innerWidth, window.innerHeight);
}

export function addCheckerboard(scene, boardSize, squareSize, offsetZ = 5) {
  checkerboardMaterial1 = new THREE.MeshStandardMaterial({
    color: 0x808080,
    side: THREE.DoubleSide,
    opacity: 0.0,
    transparent: false,
  });
  checkerboardMaterial2 = new THREE.MeshStandardMaterial({
    color: 0x505050,
    side: THREE.DoubleSide,
    opacity: 0.0,
    transparent: false,
  });
  checkerboardShadowMaterial = new THREE.ShadowMaterial({
    opacity: 0.25,
  });
  checkerboardGroup = new THREE.Group();
  for (let i = 0; i < boardSize; i++) {
    for (let j = 0; j < boardSize; j++) {
      const squareGeometry = new THREE.PlaneGeometry(squareSize, squareSize);
      const squareMaterial = (i + j) % 2 === 0 ? checkerboardMaterial1 : checkerboardMaterial2;
      const square = new THREE.Mesh(squareGeometry, squareMaterial);
      square.position.x = i * squareSize - (boardSize * squareSize) / 2;
      square.position.y = 0;
      square.position.z = j * squareSize - (boardSize * squareSize) / 2 - offsetZ;
      square.rotation.x = -Math.PI / 2;
      square.receiveShadow = true;
      checkerboardGroup.add(square);
    }
  }
  scene.add(checkerboardGroup);
  return checkerboardGroup;
}

export function updateCheckerboard(xr) {
  checkerboardGroup.children.forEach((square, index) => {
    if (xr) {
      square.material = checkerboardShadowMaterial;
    } else {
      square.material = index % 2 === 0 ? checkerboardMaterial1 : checkerboardMaterial2;
    }
  });
}

export async function fetchRangeURL(url, start, end) {
  const response = await fetch(url, { headers: { 'Range': `bytes=${start}-${end}` } });
  if (!response.ok) {
      throw new Error(`Erreur HTTP : ${response.status}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  return arrayBuffer;
}

export async function fetchFramePKL(i) {
  try {
    const response = await fetch("/get?frame=" + i.toString());
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get products: ${error}`);
  }
}

export async function fetchFramePT(i) {
  try {
    const response = await fetch("/get?frame=" + i.toString());
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get products: ${error}`);
  }
}

export async function fetchVerticesPKL(i) {
  try {
    const response = await fetch("/getVertices?frame=" + i.toString());
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get products: ${error}`);
  }
}

export async function fetchInfosPKL() {
  try {
    const response = await fetch("/getInfos");
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get response: ${error}`);
  }
}

export async function fetchInfosPT() {
  try {
    const response = await fetch("/getInfos");
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get response: ${error}`);
  }
}

export async function fetchInfosURL(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`File error: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Could not get response: ${error}`);
  }
}

// --- Track functions ---
// Modified addTrack uses the provided poseId.
function addTrack(track, poseId, index, allTracksGroup) {
  const trackGeometry = new THREE.BufferGeometry();
  const trackPositions = new Float32Array(track.length * 3);
  for (let i = 0; i < track.length; i++) {
    trackPositions[i * 3]     = track[i][index + 2][0];
    trackPositions[i * 3 + 1] = track[i][index + 2][1];
    trackPositions[i * 3 + 2] = track[i][index + 2][2];
  }
  trackGeometry.setAttribute("position", new THREE.BufferAttribute(trackPositions, 3));
  const colorId = idColors[(poseId + 1) % idColors.length];
  const trackMaterial = new THREE.PointsMaterial({ size: 0.008, color: colorId });
  const trackPoints = new THREE.Points(trackGeometry, trackMaterial);
  // Store the pose id for later use.
  trackPoints.userData.poseId = poseId;
  allTracksGroup.add(trackPoints);
}

// Modified addAllTracks now expects each element of allTracks to be an object with keys "poseId" and "data".
export function addAllTracks(allTracks, index, allTracksGroup, params, scene) {
  if (allTracksGroup !== null) scene.remove(allTracksGroup);
  for (let i = 0; i < allTracks.length; i++) {
    const trackObj = allTracks[i];
    if (trackObj.data && trackObj.data.length > 1) {
      const poseId = trackObj.poseId;
      addTrack(trackObj.data, poseId, index, allTracksGroup);
    }
  }
  allTracksGroup.rotation.x = Math.PI - (params.cameraRotationX * Math.PI) / 180;
  allTracksGroup.rotation.z = (params.cameraRotationY * Math.PI) / 180;
  allTracksGroup.position.y = 0;
  // allTracksGroup.position.y = params.globalZOffset;
  scene.add(allTracksGroup);
}
// --- Body functions ---
function addBody(vertices, id, bodiesArray, allBodiesGroup, smplx_faces, scene) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(vertices);
  const colorId = idColors[(id + 1) % idColors.length];
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(smplx_faces);
  const material = new THREE.MeshLambertMaterial({ color: colorId, opacity: 0.5, transparent: true, wireframe: false });
  const body = new THREE.Mesh(geometry, material);
  body.castShadow = true;
  body.receiveShadow = false;
  body.visible = false;
  bodiesArray.push(body);
  allBodiesGroup.add(body);
}

export function initBodies(maxHumans, allBodiesGroup, params, scene, 
  bodiesArrayWithId, bodiesArrayWithoutId, smplx_faces) {
  const N = 10475;
  for (let i = 0; i < maxHumans; i++) {
    const vertices = new Float32Array(N * 3);
    addBody(vertices, 1, bodiesArrayWithId, allBodiesGroup, smplx_faces, scene);
    addBody(vertices, -1, bodiesArrayWithoutId, allBodiesGroup, smplx_faces, scene);
  }
  allBodiesGroup.rotation.x = Math.PI - (params.cameraRotationX * Math.PI) / 180;
  allBodiesGroup.rotation.z = (params.cameraRotationY * Math.PI) / 180;
  allBodiesGroup.position.y = 0;
  // allBodiesGroup.position.y = params.globalZOffset;
  scene.add(allBodiesGroup);
}

function updateBody(vertices, humanId, body) {
  body.geometry.attributes.position.array.set(vertices);
  body.geometry.attributes.position.needsUpdate = true;
  body.material.color.setHex(idColors[(humanId + 1) % idColors.length]); 
}

function updateBodyWithNormals(vertices, normals, humanId, body) 
{
  const vec = new THREE.Vector3( normals[0], normals[1], normals[2] );
  body.geometry.attributes.normal.array.set(normals);
  body.geometry.attributes.normal.needsUpdate = true;
  body.geometry.attributes.position.array.set(vertices);
  body.geometry.attributes.position.needsUpdate = true;
  body.material.color.setHex(idColors[(humanId + 1) % idColors.length]); 
}

export function updateBodiesWithNormals(allHumenVertices, allHumenNormals, allId, maxHumans, bodiesArrayWithId, bodiesArrayWithoutId ) {
  // First hide all bodies
  for (let i = 0; i < maxHumans; i++) {
    bodiesArrayWithoutId[i].visible = false;
    bodiesArrayWithId[i].visible = false;
  }
  
  // Then process each body based on its ID and visibility settings
  for (let i = 0; i < allId.length; i++) {
    const humanId = allId[i];
    
    // Check if this track/pose should be visible based on track visibility settings
    // Default to visible for unidentified poses (-1)
    let isVisible = true;
    if (humanId !== -1) {
      isVisible = (window.trackVisibility && window.trackVisibility[humanId] !== undefined) ? 
                 window.trackVisibility[humanId] : true;
    }
    
    if (isVisible) {
      if (humanId === -1) {
        // Handle bodies without IDs
        bodiesArrayWithoutId[i].visible = true;
        bodiesArrayWithId[i].visible = false;
        // Update the mesh with vertices and normals
        if (allHumenVertices && allHumenNormals) {
          bodiesArrayWithoutId[i].geometry.attributes.normal.array.set(allHumenNormals[i]);
          bodiesArrayWithoutId[i].geometry.attributes.normal.needsUpdate = true;
          bodiesArrayWithoutId[i].geometry.attributes.position.array.set(allHumenVertices[i]);
          bodiesArrayWithoutId[i].geometry.attributes.position.needsUpdate = true;
          bodiesArrayWithoutId[i].material.color.setHex(idColors[(humanId + 1) % idColors.length]);
        }
      } else {
        // Handle bodies with IDs
        bodiesArrayWithoutId[i].visible = false;
        bodiesArrayWithId[i].visible = true;
        // Update the mesh with vertices and normals
        if (allHumenVertices && allHumenNormals) {
          bodiesArrayWithId[i].geometry.attributes.normal.array.set(allHumenNormals[i]);
          bodiesArrayWithId[i].geometry.attributes.normal.needsUpdate = true;
          bodiesArrayWithId[i].geometry.attributes.position.array.set(allHumenVertices[i]);
          bodiesArrayWithId[i].geometry.attributes.position.needsUpdate = true;
          bodiesArrayWithId[i].material.color.setHex(idColors[(humanId + 1) % idColors.length]);
        }
      }
    }
  }
}

export function updateBodies(allHumenVertices, allId, maxHumans, bodiesArrayWithId, bodiesArrayWithoutId) {
  // First hide all bodies
  for (let i = 0; i < maxHumans; i++) {
    bodiesArrayWithoutId[i].visible = false;
    bodiesArrayWithId[i].visible = false;
  }
  
  // Then process each body based on its ID and visibility settings
  for (let i = 0; i < allId.length; i++) {
    const humanId = allId[i];
    
    // Check if this track/pose should be visible based on track visibility settings
    // Default to visible for unidentified poses (-1)
    let isVisible = true;
    if (humanId !== -1) {
      isVisible = (window.trackVisibility && window.trackVisibility[humanId] !== undefined) ? 
                 window.trackVisibility[humanId] : true;
    }
    
    if (isVisible) {
      if (humanId === -1) {
        bodiesArrayWithoutId[i].visible = true;
        bodiesArrayWithId[i].visible = false;
        updateBody(allHumenVertices[i], humanId, bodiesArrayWithoutId[i]);
      } else {
        bodiesArrayWithoutId[i].visible = false;
        bodiesArrayWithId[i].visible = true;
        updateBody(allHumenVertices[i], humanId, bodiesArrayWithId[i]);
      }
    }
  }
}

export function updateBodiesPKL(allData, maxHumans, bodiesArrayWithId, bodiesArrayWithoutId) {
  const allHumenVertices = allData["2"];
  const allId = allData["1"];
  updateBodies(allHumenVertices, allId, maxHumans, bodiesArrayWithId, bodiesArrayWithoutId);
}

// --- Skeleton functions ---
function addSkeleton(keypoints, id, skeletonsArray, allSkeletonsGroup) {
  const skeleton = new THREE.Group();
  const sphereMaterial = new THREE.MeshLambertMaterial(idColors[(id + 1) % idColors.length]);
  const spheres = new THREE.Group();
  for (let i = 0; i < keypoints.length; i++) {
    const radius = (i < 25) ? 0.020 : 0.005;
    const sphereGeometry = new THREE.SphereGeometry(radius, 5, 5);
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(keypoints[i][0], keypoints[i][1], keypoints[i][2]);
    sphere.castShadow = true;
    sphere.receiveShadow = false;
    spheres.add(sphere);
  }
  skeleton.add(spheres);
  const bones = new THREE.Group();
  const bonesMaterial = new THREE.MeshLambertMaterial(idColors[(id + 1) % idColors.length]);
  for (let i = 0; i < connections.length; i++) {
    const boneGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1);
    const bone = new THREE.Mesh(boneGeometry, bonesMaterial);
    bone.castShadow = true;
    bone.receiveShadow = false;
    bones.add(bone);
  }
  skeleton.add(bones);
  skeleton.visible = false;
  skeletonsArray.push(skeleton);
  allSkeletonsGroup.add(skeleton);
}

export function initSkeletons(maxHumans, allSkeletonsGroup, params, scene, 
  skeletonsArrayWithId, skeletonsArrayWithoutId, nbKeypoints = 127) {
  const keypoints = [];
  for (let i = 0; i < nbKeypoints; i++) 
      keypoints.push([0, 0, 0]);
  for (let i = 0; i < maxHumans; i++) {
      addSkeleton(keypoints, 1, skeletonsArrayWithId, allSkeletonsGroup);
      addSkeleton(keypoints, -1, skeletonsArrayWithoutId, allSkeletonsGroup);
  }
  allSkeletonsGroup.rotation.x = Math.PI - (params.cameraRotationX * Math.PI) / 180;
  allSkeletonsGroup.rotation.z = (params.cameraRotationY * Math.PI) / 180;
  allSkeletonsGroup.position.y = 0;
  scene.add(allSkeletonsGroup);
}

function updateSkeleton(keypoints, humanId, skeleton, nbKeypoints) {
  for (let i = 0; i < nbKeypoints; i++) {
    skeleton.children[0].children[i].position.set(keypoints[i][0], keypoints[i][1], keypoints[i][2]);
  }
  for (let i = 0; i < connections.length; i++) {
    const bone = skeleton.children[1].children[i];
    bone.material.color.setHex(idColors[(humanId + 1) % idColors.length]);
    const p1 = new THREE.Vector3(...keypoints[connections[i][0]]);
    const p2 = new THREE.Vector3(...keypoints[connections[i][1]]);
    let actualHeight = p1.distanceTo(p2);
    let midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    let direction = new THREE.Vector3().subVectors(p2, p1).normalize();
    bone.position.copy(midPoint);
    bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    bone.scale.set(1, actualHeight, 1);
  }
}

export function updateSkeletons(allHumenKeypoints, allId, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId, nbKeypoints = 127) {
  for (let i = 0; i < maxHumans; i++) {
    skeletonsArrayWithoutId[i].visible = false;
    skeletonsArrayWithId[i].visible = false;
  }
  for (let i = 0; i < allId.length; i++) {
    // Use the actual pose ID (allId[i]) to check visibility instead of array index (i)
    let poseId = allId[i];
    // Default to true for unidentified poses (-1)
    let toggle;
    if (poseId === -1) {
      toggle = true;
    } else {
      toggle = (poseId === -1) || (window.trackVisibility && window.trackVisibility[poseId] !== undefined) ? 
                 window.trackVisibility[poseId] : true;
    }
    if (toggle) {
      if (poseId === -1) {
        skeletonsArrayWithoutId[i].visible = true;
        skeletonsArrayWithId[i].visible = false;
        updateSkeleton(allHumenKeypoints[i], poseId, skeletonsArrayWithoutId[i], nbKeypoints);
      } else {
        skeletonsArrayWithoutId[i].visible = false;
        skeletonsArrayWithId[i].visible = true;
        updateSkeleton(allHumenKeypoints[i], poseId, skeletonsArrayWithId[i], nbKeypoints);
      }
    }
  }
}

export function updateSkeletonsPKL(allData, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId, nbKeypoints = 127) {
  const allHumenKeypoints = allData["0"];
  const allId = allData["1"];
  updateSkeletons(allHumenKeypoints, allId, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId, nbKeypoints);
}

// --- Skeleton PT functions ---
function addSkeletonPT(keypoints, id, skeletonsArray, allSkeletonsGroup) {
  const skeleton = new THREE.Group();
  const sphereMaterial = new THREE.MeshLambertMaterial(idColors[(id + 1) % idColors.length]);
  const spheres = new THREE.Group();
  for (let i = 0; i < 127; i++) {
      const sphereGeometry = new THREE.SphereGeometry(0.015, 5, 5);
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere.position.set(keypoints[i][0], keypoints[i][1], keypoints[i][2]);
      sphere.castShadow = true;
      sphere.receiveShadow = false;
      spheres.add(sphere);
  }
  skeleton.add(spheres);
  const bones = new THREE.Group();
  const bonesMaterial = new THREE.MeshLambertMaterial(idColors[(id + 1) % idColors.length]);
  for (let i = 0; i < connectionsPT.length; i++) {
    const boneGeometry = new THREE.CylinderGeometry(0.005, 0.005, 1);
    const bone = new THREE.Mesh(boneGeometry, bonesMaterial);
    bone.castShadow = true;
    bone.receiveShadow = false;
    bones.add(bone);
  }
  skeleton.add(bones);
  skeleton.visible = false;
  skeletonsArray.push(skeleton);
  allSkeletonsGroup.add(skeleton);
}

export function initSkeletonsPT(maxHumans, allSkeletonsGroup, params, scene, 
  skeletonsArrayWithId, skeletonsArrayWithoutId) {
  const keypoints = [];
  for (let i = 0; i < 25; i++) 
    keypoints.push([0, 0, 0]);
  for (let i = 0; i < maxHumans; i++) {
      addSkeletonPT(keypoints, 1, skeletonsArrayWithId, allSkeletonsGroup);
      addSkeletonPT(keypoints, -1, skeletonsArrayWithoutId, allSkeletonsGroup);
  }
  allSkeletonsGroup.rotation.x = (params.cameraRotationX * Math.PI) / 180;
  allSkeletonsGroup.rotation.z = (params.cameraRotationY * Math.PI) / 180;
  allSkeletonsGroup.position.y = 0;
  // allSkeletonsGroup.position.y = params.globalZOffset;
  scene.add(allSkeletonsGroup);
}

function updateSkeletonPT(keypoints, humanId, skeleton) {
  for (let i = 0; i < keypoints.length; i++)
    skeleton.children[0].children[i].position.set(keypoints[i][0], keypoints[i][1], keypoints[i][2]);
  for (let i = 0; i < connectionsPT.length; i++) {
    const bone = skeleton.children[1].children[i];
    bone.material.color.setHex(idColors[(humanId + 1) % idColors.length]);
    const p1 = new THREE.Vector3(...keypoints[connectionsPT[i][0]]);
    const p2 = new THREE.Vector3(...keypoints[connectionsPT[i][1]]);
    let actualHeight = p1.distanceTo(p2);
    let midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    let direction = new THREE.Vector3().subVectors(p2, p1).normalize();
    bone.position.copy(midPoint);
    bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    bone.scale.set(1, actualHeight, 1);
  }
}

export function updateSkeletonsBasePT(allHumenKeypoints, allId, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId) {
  for (let i = 0; i < maxHumans; i++) {
    skeletonsArrayWithoutId[i].visible = false;
    skeletonsArrayWithId[i].visible = false;
  }
  for (let i = 0; i < allId.length; i++) {
    if (allId[i] === -1) {
      skeletonsArrayWithoutId[i].visible = true;
      skeletonsArrayWithId[i].visible = false;
      updateSkeletonPT(allHumenKeypoints[i], allId[i], skeletonsArrayWithoutId[i]);
    } else {
      skeletonsArrayWithoutId[i].visible = false;
      skeletonsArrayWithId[i].visible = true;
      updateSkeletonPT(allHumenKeypoints[i], allId[i], skeletonsArrayWithId[i]);
    }
  }
}

export function updateSkeletonsPT(allData, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId) {
  const allHumenKeypoints = allData["0"];
  const allId = allData["1"];
  updateSkeletonsBasePT(allHumenKeypoints, allId, maxHumans, skeletonsArrayWithId, skeletonsArrayWithoutId);
}
