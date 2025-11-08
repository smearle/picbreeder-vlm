/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.utilities;

import org.w3c.dom.*;

import client.*;
import java.io.*;
import java.util.zip.ZipFile;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

/**
 * Utilities for parsing XML DOM objects.
 * 
 * @author Nick Beato
 */

public class XML {
	private static final String VERSION = "1.0";
	
	/**
	 * Parses the text content of a child element named
	 * <code>elementName</code>.
	 * 
	 * @param xmlElement The parent element
	 * @param elementName The name of the child element to parse
	 * @return The text content of the element
	 */
	
	public static String parseElementText(Element xmlElement, String elementName) {
		return xmlElement.getElementsByTagName(elementName).item(0).getTextContent();
	}
	
	
	/**
	 * Create an element name <code>childName</code> with the text content
	 * <code>value</code> using the <code>xmlDocument</code> as a factory.
	 * 
	 * @param childName The name of the created element
	 * @param value The value of the created element's text field
	 * @param xmlDocument The factory
	 * @return A new element with the specified information
	 */
	public static Element createElementText(String childName, String value, Document xmlDocument) {
		Element child = xmlDocument.createElement(childName);
		child.setTextContent(value);
		return child;
	}
	
	/**
	 * Creates an element for the transferable object and stores the object
	 * in it.
	 * 
	 * @param object The object
	 * @param parentElement The parent element
	 * @param xmlDocument The factory
	 */
	public static void storeElement(Transferable object, Element parentElement, Document xmlDocument) {
		Element element = xmlDocument.createElement(object.getElementName());
		object.store(element, xmlDocument);
		parentElement.appendChild(element);
	}
	
	/**
	 * Loads the first child element of parentElement that can be loaded by the
	 * transferable object into the object.
	 * 
	 * @param object The object to load
	 * @param rootElement The element tree
	 */
	public static void loadElement(Transferable object, Element rootElement) {
		object.load((Element) rootElement.getElementsByTagName(object.getElementName()).item(0));
	}
	
	/**
	 * Saves a Transferable to a an xml file.
	 */
	public static void store(Transferable object, File file)
			throws IOException {
		
		// ewwwwwww
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
			
			Element root = doc.createElement(object.getElementName());
			doc.appendChild(root);
			object.store(root, doc);
			
			TransformerFactory transFactory = TransformerFactory.newInstance();
			transFactory.setAttribute("indent-number", 4);
			Transformer trans = transFactory.newTransformer();
			trans.setOutputProperty(OutputKeys.INDENT, "yes");
			trans.transform(new DOMSource(doc), new StreamResult(new OutputStreamWriter(new FileOutputStream(file))));
			
			/*
			// WORKS
			ByteArrayOutputStream writer = new ByteArrayOutputStream();
			trans.transform(new DOMSource(doc), new StreamResult(new OutputStreamWriter(writer)));
			writer.close();
			
			java.util.zip.ZipOutputStream zip = new java.util.zip.ZipOutputStream(new BufferedOutputStream(new FileOutputStream(new File(fileName + ".zip"))));
			zip.putNextEntry(new java.util.zip.ZipEntry(fileName));
			zip.write(writer.toByteArray());
			zip.close();
			*/
		}
		catch(IOException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("XML Errors");
		}
	}
	
	public static void store(Transferable object, OutputStream writer)
			throws IOException {
	
		// ewwwwwww
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
			
			Element root = doc.createElement(object.getElementName());
			
			Element realRoot = doc.createElement("data");
			
			// hack to insert a parameter table id... may be used in the future
			if(object instanceof client.evolution.Series) {
				Element ps = doc.createElement("parameters");
				ps.setTextContent("default");
				realRoot.appendChild(ps);
			}
			
			realRoot.setAttribute("version", VERSION);
			realRoot.appendChild(root);
			
			doc.appendChild(realRoot);
			object.store(root, doc);
			
			TransformerFactory transFactory = TransformerFactory.newInstance();
			transFactory.setAttribute("indent-number", 4);
			Transformer trans = transFactory.newTransformer();
			trans.setOutputProperty(OutputKeys.INDENT, "yes");
			trans.transform(new DOMSource(doc), new StreamResult(new OutputStreamWriter(writer)));
			//writer.close();
			/*
			java.util.zip.ZipOutputStream zip = new java.util.zip.ZipOutputStream(new BufferedOutputStream(new FileOutputStream(new File(fileName + ".zip"))));
			zip.putNextEntry(new java.util.zip.ZipEntry(fileName));
			zip.write(writer.toByteArray());
			zip.close();
			*/
		}
		/*catch(IOException e) {
			throw e;
		}*/
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("XML Errors");
		}
	}
	
	/**
	 * Loads a Transferable from a file.
	 */
	public static void loadFromFile(Transferable object, String fileName) throws IOException {
		// If the filename ends with .zip, unzip and then render.  Render normally otherwise
		if(fileName.lastIndexOf(".zip") >= 0) {
			ZipFile zipfile = new ZipFile(fileName);
			InputStream s = new BufferedInputStream(zipfile.getInputStream(zipfile.entries().nextElement()));
			XML.load(object, s);
			s.close();
		}
		else {
			InputStream s = new BufferedInputStream(new FileInputStream(new File(fileName)));
			load(object, s);
			s.close();
		}
	}
	
	public static void load(Transferable object, InputStream stream) {
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(stream);
			Element root = doc.getDocumentElement();
			
			if(root.getNodeName().equals("data")) {
				Node n = root.getFirstChild();
				while(n != null) {
					// TODO hack for parameter table
					if(!(n instanceof Element) || n.getNodeName().equals("parameters"))
						n = n.getNextSibling();
					else
						break;
				}
				
				if(n != null)
					root = (Element) n;
			}
			
			object.load(root);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void loadConfiguration(Configurable config, InputStream stream) {
		try {
			Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(stream);
			Element root = doc.getDocumentElement();
			
			Node n = root.getFirstChild();
			while(n != null) {
				if(n instanceof Element)
					break;
				else
					n = n.getNextSibling();
			}
			
			if(n != null)			
				config.configure((Element) n);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void store(Transferable object, String fileName)
			throws IOException {
		store(object, new File(fileName));
	}
	
	public static Document validate(String file, String dtdFolder) throws Exception {
		if(!dtdFolder.endsWith("\\") && !dtdFolder.endsWith("/"))
			dtdFolder = dtdFolder + "/";
		
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		factory.setValidating(true);
		factory.setIgnoringElementContentWhitespace(true);
		
		CustomErrorHandler handler = new CustomErrorHandler();
		
		DocumentBuilder builder = factory.newDocumentBuilder();
		builder.setErrorHandler(handler);
		Document doc = builder.parse(new java.io.FileInputStream(new java.io.File(file)), dtdFolder);
		
		if(handler.hasError())
			return null;
		else
			return doc;
	}
	
	private final static class CustomErrorHandler implements org.xml.sax.ErrorHandler {
		private boolean hasError = false;
		
		public boolean hasError() {
			return hasError;
		}
		
		public void fatalError(org.xml.sax.SAXParseException e) {
			e.printStackTrace();
			hasError = true;
		}
		
		public void error(org.xml.sax.SAXParseException e) {
			e.printStackTrace();
			hasError = true;
		}
		
		public void warning(org.xml.sax.SAXParseException e) {
			e.printStackTrace();
		}
	}
}
